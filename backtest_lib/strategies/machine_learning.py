class MachineLearningStrategy(Strategy):
    """
    Machine Learning Strategy Implementation
    
    Use ML models to predict price direction or optimize entry/exit points
    """
    
    def __init__(self, name: str = "MachineLearning", model_type: str = "lstm", lookback_period: int = 20, prediction_horizon: int = 1, features: List[str] = None):
        """
        Initialize the strategy
        
        Parameters:
        -----------
        name : str
            Strategy name
        model_type : str
            Type of ML model ('lstm', 'gru', 'xgboost')
        lookback_period : int
            Number of past periods to use for features
        prediction_horizon : int
            Number of periods to predict ahead
        features : List[str], optional
            List of features to use for prediction
        """
        super().__init__(name)
        
        if features is None:
            features = ['close', 'volume', 'rsi', 'macd', 'bb_width']
        
        self.set_parameters(
            model_type=model_type,
            lookback_period=lookback_period,
            prediction_horizon=prediction_horizon,
            features=features
        )
        
        self.model = None
        self.scaler = StandardScaler()
    
    def _create_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Create features for ML model
        
        Parameters:
        -----------
        data : pd.DataFrame
            Market data with OHLCV
            
        Returns:
        --------
        pd.DataFrame
            DataFrame with features
        """
        df = data.copy()
        
        # Price-based features
        df['returns'] = df['close'].pct_change()
        df['log_returns'] = np.log(df['close'] / df['close'].shift(1))
        
        # Technical indicators
        df['rsi'] = Indicators.rsi(df['close'])
        df['macd'], df['signal'], df['hist'] = Indicators.macd(df['close'])
        
        # Volatility features
        df['upper_band'], df['middle_band'], df['lower_band'] = Indicators.bollinger_bands(df['close'])
        df['bb_width'] = (df['upper_band'] - df['lower_band']) / df['middle_band']
        df['atr'] = Indicators.atr(df['high'], df['low'], df['close'])
        
        # Volume features
        df['volume_ma'] = Indicators.sma(df['volume'], 14)
        df['volume_ratio'] = df['volume'] / df['volume_ma']
        
        # Trend features
        df['sma_5'] = Indicators.sma(df['close'], 5)
        df['sma_20'] = Indicators.sma(df['close'], 20)
        df['sma_ratio'] = df['sma_5'] / df['sma_20']
        
        # Target variable: price direction
        df['target'] = np.where(df['close'].shift(-self.parameters['prediction_horizon']) > df['close'], 1, 0)
        
        return df
    
    def _prepare_data(self, df: pd.DataFrame, train_size: float = 0.8) -> Tuple:
        """
        Prepare data for ML model
        
        Parameters:
        -----------
        df : pd.DataFrame
            DataFrame with features
        train_size : float
            Proportion of data to use for training
            
        Returns:
        --------
        Tuple
            X_train, X_test, y_train, y_test
        """
        features = self.parameters['features']
        lookback = self.parameters['lookback_period']
        
        # Drop rows with NaN values
        df.dropna(inplace=True)
        
        # Select features
        X = df[features].values
        y = df['target'].values
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Create sequences for LSTM/GRU
        if self.parameters['model_type'] in ['lstm', 'gru']:
            X_seq = []
            y_seq = []
            
            for i in range(lookback, len(X_scaled)):
                X_seq.append(X_scaled[i-lookback:i])
                y_seq.append(y[i])
            
            X_seq = np.array(X_seq)
            y_seq = np.array(y_seq)
            
            # Split data into train and test sets
            split_idx = int(len(X_seq) * train_size)
            X_train, X_test = X_seq[:split_idx], X_seq[split_idx:]
            y_train, y_test = y_seq[:split_idx], y_seq[split_idx:]
        else:
            # For XGBoost, create lagged features
            X_lagged = np.zeros((len(X_scaled) - lookback, lookback * len(features)))
            
            for i in range(lookback, len(X_scaled)):
                row = X_scaled[i-lookback:i].flatten()
                X_lagged[i-lookback] = row
            
            y_lagged = y[lookback:]
            
            # Split data into train and test sets
            split_idx = int(len(X_lagged) * train_size)
            X_train, X_test = X_lagged[:split_idx], X_lagged[split_idx:]
            y_train, y_test = y_lagged[:split_idx], y_lagged[split_idx:]
        
        return X_train, X_test, y_train, y_test
    
    def _build_model(self, input_shape: Tuple) -> Any:
        """
        Build ML model
        
        Parameters:
        -----------
        input_shape : Tuple
            Shape of input data
            
        Returns:
        --------
        Any
            ML model
        """
        model_type = self.parameters['model_type']
        
        if model_type == 'lstm':
            model = Sequential([
                LSTM(50, return_sequences=True, input_shape=input_shape),
                Dropout(0.2),
                LSTM(50),
                Dropout(0.2),
                Dense(1, activation='sigmoid')
            ])
            
            model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
            
        elif model_type == 'gru':
            model = Sequential([
                GRU(50, return_sequences=True, input_shape=input_shape),
                Dropout(0.2),
                GRU(50),
                Dropout(0.2),
                Dense(1, activation='sigmoid')
            ])
            
            model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
            
        elif model_type == 'xgboost':
            model = xgb.XGBClassifier(
                n_estimators=100,
                max_depth=5,
                learning_rate=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
                objective='binary:logistic',
                eval_metric='logloss'
            )
        
        else:
            raise ValueError(f"Unsupported model type: {model_type}")
        
        return model
    
    def train_model(self, data: pd.DataFrame, epochs: int = 50, batch_size: int = 32, early_stopping: bool = True, verbose: int = 1) -> None:
        """
        Train ML model
        
        Parameters:
        -----------
        data : pd.DataFrame
            Market data with OHLCV
        epochs : int
            Number of training epochs
        batch_size : int
            Batch size for training
        early_stopping : bool
            Whether to use early stopping
        verbose : int
            Verbosity level
        """
        # Create features
        df = self._create_features(data)
        
        # Prepare data
        X_train, X_test, y_train, y_test = self._prepare_data(df)
        
        # Build model
        if self.parameters['model_type'] in ['lstm', 'gru']:
            input_shape = (X_train.shape[1], X_train.shape[2])
        else:
            input_shape = X_train.shape[1]
        
        self.model = self._build_model(input_shape)
        
        # Train model
        if self.parameters['model_type'] in ['lstm', 'gru']:
            callbacks = []
            
            if early_stopping:
                callbacks.append(EarlyStopping(patience=10, restore_best_weights=True))
            
            self.model.fit(
                X_train, y_train,
                validation_data=(X_test, y_test),
                epochs=epochs,
                batch_size=batch_size,
                callbacks=callbacks,
                verbose=verbose
            )
        else:
            self.model.fit(
                X_train, y_train,
                eval_set=[(X_test, y_test)],
                early_stopping_rounds=10 if early_stopping else None,
                verbose=verbose
            )
        
        logger.info(f"Model '{self.name}' trained successfully")
    
    def save_model(self, directory: str = 'saved_models') -> None:
        """
        Save ML model
        
        Parameters:
        -----------
        directory : str
            Directory to save the model
        """
        os.makedirs(directory, exist_ok=True)
        file_path = os.path.join(directory, f"{self.name}_model")
        
        if self.parameters['model_type'] in ['lstm', 'gru']:
            self.model.save(file_path)
        else:
            with open(f"{file_path}.pkl", 'wb') as f:
                pickle.dump(self.model, f)
        
        # Save scaler
        with open(os.path.join(directory, f"{self.name}_scaler.pkl"), 'wb') as f:
            pickle.dump(self.scaler, f)
        
        logger.info(f"Model '{self.name}' saved to {file_path}")
    
    def load_model(self, directory: str = 'saved_models') -> None:
        """
        Load ML model
        
        Parameters:
        -----------
        directory : str
            Directory containing the model
        """
        file_path = os.path.join(directory, f"{self.name}_model")
        
        if self.parameters['model_type'] in ['lstm', 'gru']:
            self.model = load_model(file_path)
        else:
            with open(f"{file_path}.pkl", 'rb') as f:
                self.model = pickle.load(f)
        
        # Load scaler
        with open(os.path.join(directory, f"{self.name}_scaler.pkl"), 'rb') as f:
            self.scaler = pickle.load(f)
        
        logger.info(f"Model '{self.name}' loaded from {file_path}")
    
    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Generate trading signals
        
        Parameters:
        -----------
        data : pd.DataFrame
            Market data with OHLCV
            
        Returns:
        --------
        pd.DataFrame
            DataFrame with signals
        """
        if self.model is None:
            logger.error("Model not trained or loaded")
            raise ValueError("Model not trained or loaded")
        
        df = data.copy()
        
        # Create features
        df_features = self._create_features(df)
        
        # Drop rows with NaN values
        df_features.dropna(inplace=True)
        
        # Select features
        features = self.parameters['features']
        X = df_features[features].values
        
        # Scale features
        X_scaled = self.scaler.transform(X)
        
        # Create sequences for LSTM/GRU
        lookback = self.parameters['lookback_period']
        
        if self.parameters['model_type'] in ['lstm', 'gru']:
            X_seq = []
            
            for i in range(lookback, len(X_scaled)):
                X_seq.append(X_scaled[i-lookback:i])
            
            X_seq = np.array(X_seq)
            
            # Predict
            predictions = self.model.predict(X_seq)
            
            # Adjust index for predictions
            pred_index = df_features.index[lookback:]
            
        else:
            # For XGBoost, create lagged features
            X_lagged = np.zeros((len(X_scaled) - lookback, lookback * len(features)))
            
            for i in range(lookback, len(X_scaled)):
                row = X_scaled[i-lookback:i].flatten()
                X_lagged[i-lookback] = row
            
            # Predict
            predictions = self.model.predict_proba(X_lagged)[:, 1]
            
            # Adjust index for predictions
            pred_index = df_features.index[lookback:]
        
        # Create signals DataFrame
        signals = pd.DataFrame(index=pred_index)
        signals['prediction'] = predictions
        signals['signal'] = 0
        signals['position'] = 0
        
        # Buy signal: prediction above threshold
        signals.loc[signals['prediction'] > 0.7, 'signal'] = 1
        
        # Sell signal: prediction below threshold
        signals.loc[signals['prediction'] < 0.3, 'signal'] = -1
        
        # Calculate position
        signals['position'] = signals['signal'].replace(to_replace=0, method='ffill')
        
        # Merge signals with original data
        result = pd.merge(df, signals, how='left', left_index=True, right_index=True)
        
        self.signals = result
        return result