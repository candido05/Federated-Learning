#!/usr/bin/env python3
"""
Script to update bank_account_fraud_federated_learning.ipynb with:
1. Remove all emojis
2. Add missing fairness feature engineering
3. Add fairness analysis section
4. Ensure age_above_50 is properly saved and excluded from training
"""

import json
import re

# Read the notebook
notebook_path = r"c:\Users\candi\OneDrive\Desktop\Federated-Learning\Code\novo_tcc_code\bank_account_fraud_federated_learning.ipynb"

with open(notebook_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

print(f"Original notebook has {len(nb['cells'])} cells")

# Step 1: Remove ALL emojis from all cells
def remove_emojis(text):
    """Remove all emojis from text"""
    # Common emojis used in the notebook
    emoji_pattern = re.compile("["
        u"\U0001F600-\U0001F64F"  # emoticons
        u"\U0001F300-\U0001F5FF"  # symbols & pictographs
        u"\U0001F680-\U0001F6FF"  # transport & map symbols
        u"\U0001F1E0-\U0001F1FF"  # flags
        u"\U00002702-\U000027B0"
        u"\U000024C2-\U0001F251"
        u"\u2600-\u26FF"  # miscellaneous symbols
        u"\u2700-\u27BF"  # dingbats
        "]+", flags=re.UNICODE)

    # Also remove specific emoji sequences
    text = text.replace('✅', '')
    text = text.replace('❌', '')
    text = text.replace('🔧', '')
    text = text.replace('📊', '')
    text = text.replace('🚀', '')
    text = text.replace('📂', '')
    text = text.replace('⚙️', '')
    text = text.replace('📈', '')
    text = text.replace('🔍', '')
    text = text.replace('📉', '')

    return emoji_pattern.sub(r'', text)

print("\n=== Step 1: Removing emojis ===")
emoji_count = 0
for i, cell in enumerate(nb['cells']):
    if 'source' in cell and cell['source']:
        original = ''.join(cell['source'])
        cleaned = remove_emojis(original)
        if original != cleaned:
            emoji_count += 1
            cell['source'] = [cleaned]

    # Also clean outputs
    if 'outputs' in cell:
        for output in cell['outputs']:
            if 'text' in output:
                if isinstance(output['text'], list):
                    output['text'] = [remove_emojis(t) for t in output['text']]
                else:
                    output['text'] = remove_emojis(output['text'])

print(f"Cleaned {emoji_count} cells with emojis")

# Step 2: Add fairness feature engineering to cell 9 (after basic features)
print("\n=== Step 2: Adding fairness feature engineering ===")

fairness_features_code = '''# ============================================================================
# FAIRNESS FEATURE ENGINEERING
# ============================================================================
print("\\nCriando features de interação para Fairness...")

# 2.3.1 Income x Age interaction
df['income_x_age'] = df['income'] * df['customer_age']

# 2.3.2 Age group binning + Income deviation from age group mean
df['age_group'] = pd.cut(df['customer_age'],
                         bins=[0, 30, 50, 100],
                         labels=['young', 'middle', 'senior'])
income_by_age = df.groupby('age_group')['income'].transform('mean')
df['income_vs_age_group_mean'] = df['income'] - income_by_age

# 2.3.3 Employment status + Income interaction (concatenação categórica)
df['employment_income_cat'] = df['employment_status'] + '_' + \\
                               pd.qcut(df['income'], q=4, labels=['Q1','Q2','Q3','Q4']).astype(str)

# 2.3.4 Customer Age x Credit Risk Score
df['age_x_credit_risk'] = df['customer_age'] * df['credit_risk_score']

# 2.3.5 Age above 50 flag (para análise de fairness)
df['age_above_50'] = (df['customer_age'] > 50).astype(int)

# 2.3.6 Income per year of age
df['income_per_age'] = df['income'] / (df['customer_age'] + 1)

# 2.3.7 Credit risk normalizado por employment status
emp_credit_mean = df.groupby('employment_status')['credit_risk_score'].transform('mean')
emp_credit_std = df.groupby('employment_status')['credit_risk_score'].transform('std').replace(0, 1)
df['credit_risk_vs_employment'] = (df['credit_risk_score'] - emp_credit_mean) / emp_credit_std

print("Features de fairness criadas:")
print("  - income_x_age")
print("  - age_group")
print("  - income_vs_age_group_mean")
print("  - employment_income_cat")
print("  - age_x_credit_risk")
print("  - age_above_50")
print("  - income_per_age")
print("  - credit_risk_vs_employment")
'''

additional_features_code = '''# ============================================================================
# ADDITIONAL FEATURE ENGINEERING
# ============================================================================

print("\\nCriando features adicionais...")

# 2.4.1 Proporção de validações telefônicas
df['phone_validation_score'] = df['phone_home_valid'] + df['phone_mobile_valid']

# 2.4.2 Session length anomaly (comparado com média do source)
source_session_mean = df.groupby('source')['session_length_in_minutes'].transform('mean')
df['session_length_vs_source'] = df['session_length_in_minutes'] - source_session_mean

# 2.4.3 Email similarity x Email free (interação interessante)
df['email_similarity_x_free'] = df['name_email_similarity'] * df['email_is_free']

# 2.4.4 Bank months vs Address months ratio
df['bank_vs_address_months'] = df['bank_months_count'] / (df['current_address_months_count'] + 1)

# 2.4.5 Days since request log transform (para reduzir skewness)
df['days_since_request_log'] = np.log1p(df['days_since_request'])

# 2.4.6 Proposed credit limit buckets
df['credit_limit_bucket'] = pd.cut(df['proposed_credit_limit'],
                                   bins=[0, 200, 500, 1000, 2000, np.inf],
                                   labels=['very_low', 'low', 'medium', 'high', 'very_high'])

print("Features adicionais criadas:")
print("  - phone_validation_score")
print("  - session_length_vs_source")
print("  - email_similarity_x_free")
print("  - bank_vs_address_months")
print("  - days_since_request_log")
print("  - credit_limit_bucket")
'''

# Find the cell with basic feature engineering (cell 9)
for i, cell in enumerate(nb['cells']):
    if cell['cell_type'] == 'code' and 'source' in cell:
        source = ''.join(cell['source'])
        if 'device_os_frequency' in source and 'velocity_ratio_6h_4w' in source:
            print(f"Found basic feature engineering at cell {i}")
            # Replace with complete feature engineering
            new_source = source + '\n\n' + fairness_features_code + '\n\n' + additional_features_code
            cell['source'] = [new_source]
            print("Added fairness and additional features")
            break

# Step 3: Update categorical features list (cell 10)
print("\n=== Step 3: Updating categorical features list ===")

updated_categorical_code = '''# Identificar tipos de features
print("\\nIdentificando tipos de features...")

# Features categóricas base
categorical_features = []
for col in df.columns:
    if df[col].dtype == 'object' or (df[col].dtype in ['int64', 'float64'] and df[col].nunique() < 10):
        if col not in ['fraud_bool', 'month']:
            categorical_features.append(col)

# Adicionar features categóricas criadas manualmente
categorical_features_manual = ['age_group', 'employment_income_cat', 'credit_limit_bucket']
for col in categorical_features_manual:
    if col in df.columns and col not in categorical_features:
        categorical_features.append(col)

# Converter para string (necessário para encoding)
for col in categorical_features:
    df[col] = df[col].astype(str)

# Features numéricas (excluindo target, month e categóricas)
numeric_features = [col for col in df.select_dtypes(include=[np.number]).columns
                   if col not in ['fraud_bool', 'month'] and col not in categorical_features]

print(f"Features categóricas ({len(categorical_features)}): {categorical_features}")
print(f"Features numéricas ({len(numeric_features)}): {numeric_features[:10]}... (showing first 10)")
'''

for i, cell in enumerate(nb['cells']):
    if cell['cell_type'] == 'code' and 'source' in cell:
        source = ''.join(cell['source'])
        if 'Identificando tipos de features' in source and i < 15:
            print(f"Found categorical features identification at cell {i}")
            cell['source'] = [updated_categorical_code]
            print("Updated categorical features list")
            break

# Step 4: Update temporal split to save age_above_50 (cell 12)
print("\n=== Step 4: Updating temporal split to save age_above_50 ===")

updated_split_code = '''print("\\nSplit Temporal...")

# Separar por mês
train_df = df[df['month'] <= 5].copy()
val_df = df[df['month'] == 6].copy()
test_df = df[df['month'] == 7].copy()

print(f"Treino (meses 0-5): {len(train_df)} amostras | {train_df['fraud_bool'].mean()*100:.2f}% fraudes")
print(f"Validação (mês 6): {len(val_df)} amostras | {val_df['fraud_bool'].mean()*100:.2f}% fraudes")
print(f"Teste (mês 7): {len(test_df)} amostras | {test_df['fraud_bool'].mean()*100:.2f}% fraudes")

# IMPORTANTE: Guardar age_above_50 ANTES de separar features
# Esta coluna será usada APENAS para análise de fairness, NÃO para treino
print("\\nGuardando age_above_50 para análise de fairness...")
age_above_50_train_full = train_df['age_above_50'].copy()
age_above_50_val_full = val_df['age_above_50'].copy()
age_above_50_test_full = test_df['age_above_50'].copy()
print("age_above_50 salvo para treino, validação e teste")

# Separar features e target (EXCLUINDO age_above_50)
feature_cols = [c for c in df.columns if c not in ['fraud_bool', 'month', 'age_above_50']]

X_train = train_df[feature_cols]
y_train = train_df['fraud_bool']

X_val = val_df[feature_cols]
y_val = val_df['fraud_bool']

X_test = test_df[feature_cols]
y_test = test_df['fraud_bool']

print(f"\\nFeatures shape: {X_train.shape[1]} (age_above_50 EXCLUÍDA do treino)")
'''

for i, cell in enumerate(nb['cells']):
    if cell['cell_type'] == 'code' and 'source' in cell:
        source = ''.join(cell['source'])
        if 'Split Temporal' in source and 'train_df = df[df' in source:
            print(f"Found temporal split at cell {i}")
            cell['source'] = [updated_split_code]
            print("Updated temporal split to save age_above_50")
            break

print("\n=== Saving updated notebook ===")

# Save the updated notebook
with open(notebook_path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

print(f"Notebook updated successfully!")
print(f"Total cells: {len(nb['cells'])}")
print("\nNext steps:")
print("1. Manually add Section 11: Fairness Analysis")
print("2. Update partitioning to also save age_above_50 per client")
print("3. Test the notebook")
