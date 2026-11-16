"""Generate synthetic scan/label fixtures; no private documents or model downloads."""
import argparse
from pathlib import Path
from PIL import Image,ImageDraw,ImageFont
parser=argparse.ArgumentParser();parser.add_argument('destination',type=Path);root=parser.parse_args().destination;(root/'scans').mkdir(parents=True,exist_ok=True)
font=ImageFont.truetype('/System/Library/Fonts/Supplemental/Arial.ttf',28)
references=['ALPHA','BRAVO','CHARLIE','DELTA','ECHO','FOXTROT','GOLF','HOTEL','INDIA','JULIET','KILO','LIMA']
rows=['relative_path,category']
for category,kind,lines in [('Invoice','invoice',['Invoice payment total balance due','Please pay the attached invoice']),('Contract','contract',['Contract agreement legal terms signed','This agreement governs the contract']),('Support','support',['Support help login account problem','Please reset access to the account'])]:
 for index,word in enumerate(references):
  name=f'{kind}-{index+1:02d}.png';im=Image.new('RGB',(850,260),'white');draw=ImageDraw.Draw(im)
  for y,line in enumerate(lines+['Reference '+word]):draw.text((20,20+y*68),line,fill='black',font=font)
  im.save(root/'scans'/name);rows.append(name+','+category)
(root/'labels.csv').write_text('\n'.join(rows)+'\n')
(root/'unseen.csv').write_text('document_text\ninvoice payment overdue balance\ncontract agreement signed legal\nsupport help login account\n')
print(root.resolve())
