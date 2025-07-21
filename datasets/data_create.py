import glob
import os
import csv

# 処理対象のフォルダと出力ファイル
folder_path = "20250712"
input_folder = os.path.join("/home/sagemaker-user/kiwami-AI-AWS/datasets", folder_path)
output_data = os.path.join("output.csv")

# 台本文字数の閾値
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
                    ad_view_str = row.get('累計再生回数', '0').replace(',', '').strip()
                    ad_view = int(ad_view_str)  # ここで変換
                    ad_date = row.get('公開日', '')
                    #writer.writerow([ad_name, ad_view, ad_date, script])
                    writer.writerow([script, ad_view])
                    total_count += 1

print(f"{total_count} 件のデータが見つかりました。")
