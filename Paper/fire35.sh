#!/bin/bash
set -e
/usr/bin/python3 polish35.py "$1"
DS_KEY=$(sed -n 's/^Deepseek API = //p' /mnt/backup/API.txt)
OA_KEY=$(sed -n 's/^OpenAI API = //p' /mnt/backup/API.txt)
GM_KEY=$(sed -n 's/^Gemini API = //p' /mnt/backup/API.txt)
/usr/bin/python3 - <<PYEOF
import json
p = json.load(open('/tmp/p35.json'))['prompt']
json.dump({"model":"deepseek-chat","messages":[{"role":"user","content":p}],"temperature":0.2}, open('/tmp/pay_ds.json','w'))
json.dump({"model":"gpt-4o","messages":[{"role":"user","content":p}],"temperature":0.2}, open('/tmp/pay_oa.json','w'))
json.dump({"contents":[{"parts":[{"text":p}]}]}, open('/tmp/pay_gm.json','w'))
PYEOF
curl -s --max-time 400 https://api.deepseek.com/chat/completions -H "Content-Type: application/json" -H "Authorization: Bearer $DS_KEY" --data @/tmp/pay_ds.json > /tmp/p35_ds.json &
curl -s --max-time 400 https://api.openai.com/v1/chat/completions -H "Content-Type: application/json" -H "Authorization: Bearer $OA_KEY" --data @/tmp/pay_oa.json > /tmp/p35_oa.json &
curl -s --max-time 400 "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-pro:generateContent?key=$GM_KEY" -H "Content-Type: application/json" --data @/tmp/pay_gm.json > /tmp/p35_gm.json &
wait
for f in ds oa gm; do
/usr/bin/python3 - <<PYEOF
import json
d = json.load(open('/tmp/p35_$f.json'))
t = d['choices'][0]['message']['content'] if 'choices' in d else d['candidates'][0]['content']['parts'][0]['text']
open('../.aris/reviews/polish35-round$1-$f.md','w').write(t)
print('=====', '$f', '=====')
print(t[:1400])
PYEOF
done
