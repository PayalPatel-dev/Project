# ============================================================================
# MIMIC-IV PREPROCESSING PIPELINE FOR PATIENT DETERIORATION PREDICTION
# ============================================================================
# This script processes ICU vital signs data from MIMIC-IV SQLite database
# and prepares it for LSTM model training.
#
# Input:  mimic_iv.db (SQLite database with vital signs)
# Output: processed_data.npz (train/val/test splits ready for LSTM)
# ============================================================================

import pandas as pd
import numpy as np
import sqlite3
from datetime import datetime, timedelta
from scipy import stats
import warnings
import logging
from tqdm import tqdm
import os

warnings.filterwarnings('ignore')

# ============================================================================
# LOGGING CONFIGURATION
# ============================================================================

# Create logs directory if it doesn't exist
LOG_DIR = 'logs'
if not os.path.exists(LOG_DIR):
    os.makedirs(LOG_DIR)
    print(f"[*] Created logs directory: {LOG_DIR}")

# Generate timestamped log filename
TIMESTAMP = datetime.now().strftime('%Y%m%d_%H%M%S')
LOG_FILE = os.path.join(LOG_DIR, f'preprocessing_{TIMESTAMP}.log')

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)-8s | %(message)s',
    handlers=[
        logging.FileHandler(LOG_FILE, encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# ============================================================================
# MIMIC-IV PREPROCESSOR CLASS
# ============================================================================

class MIMICPreprocessor:
    """
    Complete preprocessing pipeline for MIMIC-IV vital signs data.
    
    This class handles:
    - Loading ICU stays and vital signs from SQLite
    - Resampling to hourly frequency
    - Creating sliding time windows
    - Train/validation/test splits
    """
    
    def __init__(self, db_path='mimic_iv.db'):
        """Initialize with database path"""
        self.db_path = db_path
        self.conn = None
        
    def connect(self):
        """Connect to SQLite database"""
        try:
            self.conn = sqlite3.connect(self.db_path)
            logger.info(f"✓ Connected to database: {self.db_path}")
            return True
        except Exception as e:
            logger.error(f"❌ Error connecting to database: {e}")
            return False
    
    def load_icustays(self, limit=None):
        """
        Load ICU stay information.
        
        Returns:
            DataFrame with columns: subject_id, hadm_id, stay_id, intime, outtime
        """
        query = """
        SELECT 
            i.subject_id,
            i.hadm_id,
            i.stay_id,
            i.intime,
            i.outtime,
            p.dod
        FROM icustays i
        LEFT JOIN patients p ON i.subject_id = p.subject_id
        """
        
        if limit:
            query += f" LIMIT {limit}"
        
        df = pd.read_sql_query(query, self.conn)
        
        # Convert timestamps
        df['intime'] = pd.to_datetime(df['intime'])
        df['outtime'] = pd.to_datetime(df['outtime'])
        df['dod'] = pd.to_datetime(df['dod'])
        
        # Calculate mortality within 48 hours of ICU discharge
        df['mortality_48h'] = (
            df['dod'].notna() & 
            ((df['dod'] - df['outtime']).dt.total_seconds() / 3600 <= 48)
        ).astype(int)
        
        logger.info(f"✓ Loaded {len(df)} ICU stays")
        logger.info(f"  • Unique patients: {df['subject_id'].nunique()}")
        logger.info(f"  • Mortality (48h post-discharge): {df['mortality_48h'].sum()} patients ({df['mortality_48h'].mean()*100:.1f}%)")
        
        return df
    
    def load_chartevents(self, stay_id):
        """
        Load vital signs for a specific ICU stay.
        
        Vital signs extracted:
        - Heart Rate (itemid: 220045)
        - SpO2 (itemid: 220277)
        - Respiratory Rate (itemid: 220210)
        - Systolic BP (itemid: 220050, 220179)
        - Diastolic BP (itemid: 220051, 220180)
        - Temperature (itemid: 223761, 223762)
        """
        query = """
        SELECT 
            charttime,
            itemid,
            valuenum
        FROM chartevents
        WHERE stay_id = ?
        AND itemid IN (220045, 220277, 220210, 220050, 220179, 220051, 220180, 223761, 223762)
        AND valuenum IS NOT NULL
        ORDER BY charttime
        """
        
        df = pd.read_sql_query(query, self.conn, params=(stay_id,))
        
        if df.empty:
            return None
        
        df['charttime'] = pd.to_datetime(df['charttime'])
        
        # Map itemids to vital sign names
        vital_map = {
            220045: 'heart_rate',
            220277: 'spo2',
            220210: 'resp_rate',
            220050: 'sbp',
            220179: 'sbp',
            220051: 'dbp',
            220180: 'dbp',
            223761: 'temperature',
            223762: 'temperature'
        }
        
        df['vital_sign'] = df['itemid'].map(vital_map)
        
        return df[['charttime', 'vital_sign', 'valuenum']]
    
    def resample_vitals(self, vitals_df, freq='1H'):
        """
        Resample vital signs to regular frequency (default: hourly).
        
        Uses forward-fill to handle missing values.
        """
        if vitals_df is None or vitals_df.empty:
            return None
        
        # Pivot to wide format
        vitals_pivot = vitals_df.pivot_table(
            index='charttime',
            columns='vital_sign',
            values='valuenum',
            aggfunc='mean'
        )
        
        # Resample to hourly
        vitals_resampled = vitals_pivot.resample(freq).mean()
        
        # Forward fill missing values (up to 4 hours)
        vitals_resampled = vitals_resampled.ffill(limit=4)
        
        # Backward fill remaining (for start of stay)
        vitals_resampled = vitals_resampled.bfill(limit=2)
        
        return vitals_resampled
    
    def remove_outliers(self, vitals_df):
        """
        Remove physiologically implausible values.
        """
        if vitals_df is None:
            return None
        
        # Define physiologically plausible ranges
        ranges = {
            'heart_rate': (20, 250),
            'spo2': (50, 100),
            'resp_rate': (4, 60),
            'sbp': (50, 250),
            'dbp': (20, 180),
            'temperature': (32, 42)
        }
        
        for vital, (min_val, max_val) in ranges.items():
            if vital in vitals_df.columns:
                vitals_df.loc[
                    (vitals_df[vital] < min_val) | (vitals_df[vital] > max_val),
                    vital
                ] = np.nan
        
        return vitals_df
    
    def create_sliding_windows(self, vitals_df, window_size=24, stride=1):
        """
        Create sliding time windows for LSTM input (MULTIVARIATE).
        
        Args:
            vitals_df: DataFrame with vital signs (indexed by time)
            window_size: Number of hours per window (default: 24)
            stride: Hours to slide between windows (default: 1)
        
        Returns:
            List of arrays with shape (window_size, n_features)
            where n_features = 6 vital signs:
            - heart_rate, spo2, resp_rate, sbp, dbp, temperature
        """
        if vitals_df is None or len(vitals_df) < window_size:
            return []
        
        windows = []
        
        # Define all vital signs to extract (6 total)
        vital_columns = ['heart_rate', 'spo2', 'resp_rate', 'sbp', 'dbp', 'temperature']
        
        # Check all vitals are present
        if not all(col in vitals_df.columns for col in vital_columns):
            missing = [col for col in vital_columns if col not in vitals_df.columns]
            logger.warning(f"Missing vital signs: {missing}")
            return []
        
        # Extract all vitals as multivariate array (T, 6)
        values = vitals_df[vital_columns].values
        
        for i in range(0, len(values) - window_size, stride):
            window = values[i:i+window_size]  # Shape: (window_size, 6)
            
            # Skip if too many missing values (across all vitals) - More lenient threshold
            if np.isnan(window).sum() > window_size * len(vital_columns) * 0.5:  # 50% instead of 30%
                continue
            
            # Fill remaining NaNs for each vital separately using multiple strategies
            window_df = pd.DataFrame(window, columns=vital_columns)
            
            # Step 1: Forward fill then backward fill (within-window temporal interpolation)
            window_df = window_df.ffill().bfill()
            
            # Step 2: Linear interpolation
            for col in vital_columns:
                window_df[col] = window_df[col].interpolate(method='linear', limit_direction='both')
            
            # Step 3: Aggressive median/mean imputation
            for col in vital_columns:
                if window_df[col].isnull().any():
                    # Try global median first
                    global_median = vitals_df[col].median()
                    if pd.notna(global_median):
                        window_df[col].fillna(global_median, inplace=True)
                    else:
                        # Fall back to global mean
                        global_mean = vitals_df[col].mean()
                        if pd.notna(global_mean):
                            window_df[col].fillna(global_mean, inplace=True)
            
            # Step 4: Per-column NaN check (allow up to 10% per column)
            skip_window = False
            for col in vital_columns:
                if window_df[col].isnull().sum() > window_size * 0.1:
                    skip_window = True
                    break
            
            if skip_window:
                continue
            
            # Step 5: Fill any remaining NaN with column mean in this window
            for col in vital_columns:
                if window_df[col].isnull().any():
                    col_mean = window_df[col].mean()
                    if pd.notna(col_mean):
                        window_df[col].fillna(col_mean, inplace=True)
                    else:
                        window_df[col].fillna(vitals_df[col].median(), inplace=True)
            
            # Step 6: Final check - skip only if still has NaN
            if window_df.isnull().any().any():
                continue
                
            window = window_df.values  # Shape: (window_size, 6)
            windows.append(window)
        
        return windows
    
    def preprocess(self, max_stays=140, window_size=24):
        """
        Run complete preprocessing pipeline.
        
        Steps:
        1. Load ICU stays
        2. For each stay:
           - Load vital signs
           - Resample to hourly
           - Remove outliers
           - Create sliding windows
        3. Split into train/val/test
        4. Save to processed_data.npz
        """
        logger.info("=" * 80)
        logger.info("MIMIC-IV PREPROCESSING PIPELINE STARTED")
        logger.info("=" * 80)
        
        # Connect to database
        if not self.connect():
            logger.error("Failed to connect to database")
            return
        
        # Load ICU stays
        logger.info(f"\n[STEP 1] Loading ICU stays (max: {max_stays})...")
        icustays = self.load_icustays(limit=max_stays)
        
        # Process each ICU stay
        logger.info(f"\n[STEP 2] Processing {len(icustays)} ICU stays...")
        all_windows = []
        all_labels = []
        processed_count = 0
        skipped_count = 0
        
        for idx, row in tqdm(icustays.iterrows(), total=len(icustays), desc="Processing stays"):
            stay_id = row['stay_id']
            subject_id = row['subject_id']
            mortality = row['mortality_48h']
            
            try:
                # Load vital signs
                vitals = self.load_chartevents(stay_id)
                
                if vitals is None:
                    logger.warning(f"  [Stay {stay_id}] No vital signs found")
                    skipped_count += 1
                    continue
                
                logger.debug(f"  [Stay {stay_id}] Loaded {len(vitals)} vital observations")
                
                # Resample and clean
                vitals_resampled = self.resample_vitals(vitals)
                if vitals_resampled is None:
                    logger.warning(f"  [Stay {stay_id}] Resampling failed")
                    skipped_count += 1
                    continue
                
                logger.debug(f"  [Stay {stay_id}] Resampled to {len(vitals_resampled)} hourly records")
                
                vitals_clean = self.remove_outliers(vitals_resampled)
                logger.debug(f"  [Stay {stay_id}] Outliers removed")
                
                # Create sliding windows from cleaned vitals
                windows = self.create_sliding_windows(vitals_clean, window_size=window_size)
                
                if len(windows) == 0:
                    logger.warning(f"  [Stay {stay_id}] No valid windows created")
                    skipped_count += 1
                    continue
                
                logger.info(f"  ✓ [Stay {stay_id}] Processed: {len(windows)} windows (mortality: {mortality})")
                
                # Add to dataset
                all_windows.extend(windows)
                all_labels.extend([mortality] * len(windows))
                processed_count += 1
                
            except Exception as e:
                logger.error(f"  ❌ [Stay {stay_id}] Error: {str(e)}")
                skipped_count += 1
                continue
        
        logger.info(f"\n✓ Successfully processed: {processed_count} stays")
        logger.info(f"✗ Skipped/Failed: {skipped_count} stays")
        
        # Convert to arrays
        X = np.array(all_windows)
        y = np.array(all_labels)
        
        logger.info(f"\n[STEP 3] Creating train/val/test splits...")
        logger.info(f"   Total samples: {len(X):,}")
        logger.info(f"   Positive samples: {y.sum():,}")
        logger.info(f"   Negative samples: {(1-y).sum():,}")
        
        # Shuffle
        indices = np.random.permutation(len(X))
        X = X[indices]
        y = y[indices]
        logger.info("   Data shuffled")
        
        # Split: 64% train, 16% val, 20% test
        train_end = int(0.64 * len(X))
        val_end = int(0.80 * len(X))
        
        X_train = X[:train_end]
        y_train = y[:train_end]
        X_val = X[train_end:val_end]
        y_val = y[train_end:val_end]
        X_test = X[val_end:]
        y_test = y[val_end:]
        
        logger.info(f"\n[STEP 4] Data splits (64%-16%-20%):")
        logger.info(f"   Train: {len(X_train):,} samples | Positive: {y_train.sum():,} | Negative: {len(y_train) - y_train.sum():,}")
        logger.info(f"   Val:   {len(X_val):,} samples | Positive: {y_val.sum():,} | Negative: {len(y_val) - y_val.sum():,}")
        logger.info(f"   Test:  {len(X_test):,} samples | Positive: {y_test.sum():,} | Negative: {len(y_test) - y_test.sum():,}")
        
        # Save processed data
        logger.info(f"\n[STEP 5] Saving processed data to processed_data.npz...")
        try:
            np.savez_compressed(
                'processed_data.npz',
                X_train=X_train,
                y_train=y_train,
                X_val=X_val,
                y_val=y_val,
                X_test=X_test,
                y_test=y_test
            )
            
            file_size = os.path.getsize('processed_data.npz') / (1024 * 1024)
            logger.info(f"   ✓ Saved to: processed_data.npz ({file_size:.1f} MB)")
        except Exception as e:
            logger.error(f"   ❌ Failed to save: {str(e)}")
            return
        
        logger.info("\n" + "=" * 80)
        logger.info("✓ PREPROCESSING COMPLETE!")
        logger.info("=" * 80)
        logger.info(f"Input:  mimic_iv.db")
        logger.info(f"Output: processed_data.npz ({file_size:.1f} MB)")
        logger.info(f"Log:    {LOG_FILE}")
        logger.info(f"\nNext step: Run lstm_model_for_deterioration.py")
        
        # Close connection
        if self.conn:
            self.conn.close()
            logger.info("Database connection closed")

# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    logger.info("\n" + "=" * 80)
    logger.info("MIMIC-IV PREPROCESSING PIPELINE")
    logger.info("=" * 80)
    logger.info(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"Working directory: {os.getcwd()}")
    logger.info(f"Log file: {LOG_FILE}")
    
    try:
        preprocessor = MIMICPreprocessor('mimic_iv.db')
        preprocessor.preprocess(max_stays=140, window_size=24)
    except Exception as e:
        logger.error(f"Fatal error: {str(e)}")
        raise
