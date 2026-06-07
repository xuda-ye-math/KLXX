#!/bin/bash
set -e
N=$1
/usr/bin/python3 poisson_round.py $N "$2"
DS_KEY=$(sed -n 's/^Deepseek API = //p' /mnt/backup/API.txt)
OA_KEY=$(sed -n 's/^OpenAI API = //p' /mnt/backup/API.txt)
GM_KEY=$(sed -n 's/^Gemini API = //p' /mnt/backup/API.txt)
P=$(/usr/bin/python3 -c "import json; print(json.dumps(json.load(open('/tmp/pr_round.json'))['prompt']))")
curl -s --max-time 500 https://api.deepseek.com/chat/completions -H "Content-Type: application/json" -H "Authorization: Bearer $DS_KEY" -d "{\"model\":\"deepseek-chat\",\"messages\":[{\"role\":\"user\",\"content\":$P}],\"temperature\":0.2}" > /tmp/pr_ds.json &
curl -s --max-time 500 https://api.openai.com/v1/chat/completions -H "Content-Type: application/json" -H "Authorization: Bearer $OA_KEY" -d "{\"model\":\"gpt-4o\",\"messages\":[{\"role\":\"user\",\"content\":$P}],\"temperature\":0.2}" > /tmp/pr_oa.json &
curl -s --max-time 500 "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-pro:generateContent?key=$GM_KEY" -H "Content-Type: application/json" -d "{\"contents\":[{\"parts\":[{\"text\":$P}]}]}" > /tmp/pr_gm.json &
wait
for f in ds oa gm; do
/usr/bin/python3 - <<PYEOF
import json
d = json.load(open('/tmp/pr_$f.json'))
t = d['choices'][0]['message']['content'] if 'choices' in d else d['candidates'][0]['content']['parts'][0]['text']
open('../.aris/reviews/poisson-subsection/round-$N-$f.md', 'w').write(t)
print('=====', '$f', '=====')
print(t[:1800])
PYEOF
done
