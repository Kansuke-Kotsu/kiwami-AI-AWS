#### データの成形
# 大元のデータ(DPRO_2025xxxx.csv)から、"台本"、"再生回数"、"投稿日"を抽出
# 投稿日から、日割の再生回数を計算
# 日割の再数回数を、0~100点の範囲に正規化し、CNNで学習しやすくする

import glob, os, csv
from datetime import datetime, date

# 処理対象のフォルダと出力ファイル
folder_path = "20250712"
input_folder = os.path.join("/Users/kotsukansuke/Documents/GitHub/kiwami-AI-AWS/datasets", folder_path)
#input_folder = os.path.join("/home/sagemaker-user/kiwami-AI-AWS/datasets", folder_path)
output_data = os.path.join("output_a.csv")

# 変数定義
today = date.today()
DATE_FMT = "%Y-%m-%d"
SCRIPT_LENGTH_THRESHOLD = 800
total_count = 0

# 出力ファイルのモードとヘッダー書き込み判定を設定
file_exists = os.path.exists(output_data)
mode = 'a' if file_exists else 'w'
write_header = not file_exists

with open(output_data, mode, encoding='utf-8', newline='') as f_out:
    writer = csv.writer(f_out)
    # ファイル作成時のみヘッダーを書き込む
    #if write_header:
    #writer.writerow(['商材名', '累計再生回数', '公開日', '台本'])
    writer.writerow(['text', 'score'])

    # フォルダ内の .csv ファイルをすべて取得
    pattern = os.path.join(input_folder, "*.csv")
    for input_path in glob.glob(pattern):
        print(f"Processing file: {input_path}")
        with open(input_path, 'r', encoding='utf-8', newline='') as f_in:
            reader = csv.DictReader(f_in)
            # 各行をチェック＆書き出し
            for row in reader:
                script = row.get('台本', "")
                if len(script) > SCRIPT_LENGTH_THRESHOLD:
                    ad_name = row.get('商材名', '')
                    ad_view_str = row.get('累計再生回数', '0').replace(',', '').strip() # コンマを削除して整数に変換
                    ad_view = int(ad_view_str)  # ここで変換
                    ad_date = row.get('公開日', '')
                    ad_date_fmt = datetime.strptime(ad_date, DATE_FMT).date()
                    passed_days = max((today - ad_date_fmt).days, 1)
                    ad_view_normalized = int(int(ad_view / passed_days)/100) # 日割の再生回数を100点満点に正規化
                    # 正規化されたスコアを出力ファイルに書き込む
                    writer.writerow([script, ad_view_normalized])
                    print(f"{ad_name}, Added: {ad_date}, Views: {ad_view}, Normalized Score: {ad_view_normalized}, 経過日数: {passed_days}")
                    # 累計件数をカウント
                    total_count += 1

print(f"{total_count} 件のデータが見つかりました。")
