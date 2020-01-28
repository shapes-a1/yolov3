for i in 0.6 0.7 0.8
do
  for j in 416 608
  do
    python detect_fasterrcnn.py --conf-thres $i --img-size $j --output fasterrcnn_${j}_conf${i} --source ../input_videos/client_london_police/CCTV\ footage\ of\ banned\ driver\ driving\ dangerously\ through\ London.mp4
  done
done