export WORLD_SIZE=1
python step1/DUAL-step1.py --model_path "Liuwenhao2022/RAHF-SFT" \
  --data_path "../data/ultrafeedback/rm" --output_dir "../model/DUAL/good" \
  --preference_type "chosen"

python step1/DUAL-step1.py --model_path "Liuwenhao2022/RAHF-SFT" \
  --data_path "../data/ultrafeedback/rm" --output_dir "../model/DUAL/bad" \
  --preference_type "rejected"
