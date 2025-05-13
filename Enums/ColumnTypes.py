from enum import Enum

class ColumnTypes(Enum):
    # All categorical features (including binary)
    CATEGORICAL_FEATURES = ['school', 'sex', 'address', 'famsize', 'Pstatus', 'Mjob', 'Fjob', 
                           'reason', 'guardian', 'schoolsup', 'famsup', 'paid', 'activities',
                           'nursery', 'higher', 'internet', 'romantic']
    
    # Binary features (yes/no, M/F, etc.)
    BINARY_FEATURES = ['school', 'sex', 'address', 'famsize', 'Pstatus', 'schoolsup', 'famsup', 
                      'paid', 'activities', 'nursery', 'higher', 'internet', 'romantic']
    
    # Ordinal features (numeric with ordered categories)
    ORDINAL_FEATURES = ['Medu', 'Fedu', 'traveltime', 'studytime', 'failures', 'famrel', 
                        'freetime', 'goout', 'Dalc', 'Walc', 'health']
    
    # Categorical features that need one-hot encoding (non-binary, non-ordinal)
    CATEGORICAL_FEATURES_TO_ENCODE = ['Mjob', 'Fjob', 'reason', 'guardian']
    
    # Numeric features (continuous integers)
    NUMERIC_FEATURES = ['age', 'absences']
    
    # Target variables (grades)
    TARGET_FEATURES = ['G1', 'G2', 'G3']


    

