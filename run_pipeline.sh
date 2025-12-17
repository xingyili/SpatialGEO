#!/bin/bash

set -e


current_time() {
    date "+%Y-%m-%d %H:%M:%S"
}

echo "======================================================="
echo "Starting Pipeline at $(current_time)"
echo "======================================================="
echo ""

# -------------------------------------------------------
# Ⅰ：pretrain AE (Autoencoder)
# -------------------------------------------------------
# echo "[Step 1/4] Running AE (Autoencoder)..."

# cd AE
# python main_new.py
# cd ..  # 运行完后，切回根目录，方便进入下一个文件夹

# echo ">>> AE finished successfully."
# echo ""
# sleep 2 # 暂停2秒，让你看清输出，也可以不加

# # -------------------------------------------------------
# 第二步：运行 EGAE (Graph Autoencoder)
# -------------------------------------------------------
echo "[Step 2/4] Running EGAE (Enhanced Graph Autoencoder)..."

cd EGAE
python main_new.py
cd ..

echo ">>> EGAE finished successfully."
echo ""
sleep 2

# -------------------------------------------------------
# 第三步：运行 Pretrain
# -------------------------------------------------------
echo "[Step 3/4] Running Pretrain..."

cd Pretrain
python main_new.py
cd ..

echo ">>> Pretrain finished successfully."
echo ""
sleep 2

# -------------------------------------------------------
# 第四步：运行 SpatialGEO
# -------------------------------------------------------
echo "[Step 4/4] Running SpatialGEO (Main Model)..."

cd SpatialGEO
python main_new.py
cd ..

echo "======================================================="
echo "All steps completed successfully at $(current_time)!"
echo "======================================================="