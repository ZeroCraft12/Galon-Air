import pandas as pd
import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings('ignore')

# 1. Muat Data
print("Memuat dataset...")
train_df = pd.read_csv('train.csv')
test_df = pd.read_csv('test.csv')
sample_sub = pd.read_csv('sample submission.csv')

train_df['date'] = pd.to_datetime(train_df['date'])
test_df['date'] = pd.to_datetime(test_df['date'])

# 2. Imputasi Cerdas Fitur Performa (Elo, Rank, History) yang hilang di test.csv
# Kita ambil rekam jejak TERAKHIR dari masing-masing tim di train.csv
print("Menyelamatkan fitur historis yang hilang di test.csv...")
train_df = train_df.sort_values('date')
missing_cols = [c for c in train_df.columns if c not in test_df.columns and c not in ['team_goals', 'opp_goals']]

team_history = train_df.groupby('team').last()[missing_cols].reset_index()
opponent_history = train_df.groupby('opponent').last()[missing_cols].reset_index()

# Gabungkan nilai performa terakhir ke dataset test
team_cols = [c for c in missing_cols if 'team' in c or 'h2h' in c or 'diff' in c]
test_df = pd.merge(test_df, team_history[['team'] + team_cols], on='team', how='left')

opp_cols = [c for c in missing_cols if 'opp' in c] 
test_df = pd.merge(test_df, opponent_history[['opponent'] + opp_cols], on='opponent', how='left')

# 3. Gabungkan Data Untuk Encoding yang Seragam
train_df['is_train'] = 1
test_df['is_train'] = 0
all_df = pd.concat([train_df, test_df], ignore_index=True)

# 4. Feature Engineering
print("Meracik fitur...")
all_df['year'] = all_df['date'].dt.year
all_df['month'] = all_df['date'].dt.month

cat_cols = ['gender', 'team', 'opponent', 'tournament', 'venue_country', 'confederation_team', 'confederation_opp']
for c in cat_cols:
    le = LabelEncoder()
    all_df[c] = le.fit_transform(all_df[c].astype(str))

# Perhitungan Bobot Turnamen untuk Metrik AW-MAE
train_df['tournament_str'] = train_df['tournament']
test_df['tournament_str'] = test_df['tournament']
all_df['tournament_str'] = pd.concat([train_df['tournament_str'], test_df['tournament_str']], ignore_index=True)

def get_weight(t):
    t = str(t).lower()
    if 'afc' in t:
        return 2.0
    elif 'world cup' in t and 'qualification' not in t:
        return 1.8
    elif 'friendly' in t:
        return 0.96
    else:
        return 1.20

all_df['weight'] = all_df['tournament_str'].apply(get_weight)
drop_cols = ['Id', 'match_id', 'date', 'tournament_str', 'is_train', 'team_goals', 'opp_goals']
features = [c for c in all_df.columns if c not in drop_cols]

# 5. Pelatihan Model (Poisson HistGradientBoosting)
print("Melatih model Poisson Boosting...")
train_data = all_df[all_df['is_train'] == 1].copy()
test_data = all_df[all_df['is_train'] == 0].copy()

X_train, y_team, y_opp, w_train = train_data[features], train_data['team_goals'], train_data['opp_goals'], train_data['weight']
X_test = test_data[features]

model_team = HistGradientBoostingRegressor(loss='poisson', learning_rate=0.05, max_iter=300, random_state=42)
model_team.fit(X_train, y_team, sample_weight=w_train)

model_opp = HistGradientBoostingRegressor(loss='poisson', learning_rate=0.05, max_iter=300, random_state=42)
model_opp.fit(X_train, y_opp, sample_weight=w_train)

# 6. Prediksi & Pembulatan untuk Skor Tepat (Bonus Metrik AW-MAE)
print("Membuat submisi akhir...")
# Model memprediksi angka desimal xG (Expected Goals), kita bulatkan ke integer untuk tebakan skor tepat
test_data['team_goals'] = np.round(model_team.predict(X_test)).astype(int)
test_data['opp_goals'] = np.round(model_opp.predict(X_test)).astype(int)

# Simpan Submisi
submission = test_data[['Id', 'team_goals', 'opp_goals']]
final_sub = pd.merge(sample_sub[['Id']], submission, on='Id', how='left').fillna(0).astype({'team_goals': int, 'opp_goals': int})
final_sub.to_csv('submission_gammafest_poisson.csv', index=False)
print("Berhasil! File siap diunggah ke Kaggle.")