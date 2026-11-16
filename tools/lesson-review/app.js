'use strict';
let review, stepIndex=0, mode='compare', busy=false;
const $=id=>document.getElementById(id);
function inline(text){
 const frag=document.createDocumentFragment();
 const pattern=/(\[([^\]]+)\]\(kwip-node:\/\/(?:data|model|control)\/[A-Za-z0-9_.-]+\))|(\*\*([^*]+)\*\*)|(`([^`]+)`)/g;
 let cursor=0;
 for(const m of text.matchAll(pattern)){
  frag.append(document.createTextNode(text.slice(cursor,m.index)));
  const el=document.createElement(m[1]?'span':m[3]?'strong':'code');
  el.textContent=m[2]||m[4]||m[6];
  if(m[1]){el.className='node-ref';el.title='Canvas reference · opens the block in the native lesson';}
  frag.append(el);cursor=m.index+m[0].length;
 }
 frag.append(document.createTextNode(text.slice(cursor)));return frag;
}
function rich(element,text){
 element.replaceChildren();let list=null,paragraph=null;
 for(const line of text.split('\n')){
  if(!line.trim()){list=null;paragraph=null;continue;}
  if(/^\*\*[^*]+\*\*$/.test(line)){const h=document.createElement('h3');h.textContent=line.slice(2,-2);element.append(h);list=null;paragraph=null;continue;}
  const match=line.match(/^(?:- |(\d+)\. )(.*)$/);
  if(match){const tag=match[1]?'OL':'UL';if(!list||list.tagName!==tag){list=document.createElement(tag);if(match[1])list.start=Number(match[1]);element.append(list);}const li=document.createElement('li');li.append(inline(match[2]));list.append(li);paragraph=null;}
  else{list=null;if(!paragraph){paragraph=document.createElement('p');element.append(paragraph);}else paragraph.append(document.createTextNode(' '));paragraph.append(inline(line));}
 }
}
function wordDiff(element,before,after){
 element.replaceChildren();
 if(before===after){rich(element,after);return;}
 for(const [label,text,kind] of [['− Previous wording',before,'removed-copy'],['+ Proposed wording',after,'added-copy']]){
  const section=document.createElement('section');section.className=kind;
  const heading=document.createElement('p');heading.className='diff-label';heading.textContent=label;
  const body=document.createElement('div');rich(body,text);section.append(heading,body);element.append(section);
 }
}
function render(){
 const current=review.current.steps[stepIndex],proposed=review.steps[stepIndex];
 $('steps').replaceChildren();
 if(review.steps.length>6){
  const label=document.createElement('label');label.htmlFor='section-picker';label.textContent='Review section';
  const select=document.createElement('select');select.id='section-picker';
  const groups=new Map();
  review.steps.forEach((s,i)=>{const name=s.group||'Lesson';if(!groups.has(name)){const group=document.createElement('optgroup');group.label=name;groups.set(name,group);select.append(group);}const option=document.createElement('option');option.value=String(i);option.textContent=`${i+1}. ${s.title}`;option.selected=i===stepIndex;groups.get(name).append(option);});
  select.onchange=()=>{stepIndex=Number(select.value);render();};
  const group=document.createElement('span');group.className='section-group';group.textContent=proposed.group||'';
  $('steps').append(label,select,group);
 }else{review.steps.forEach((s,i)=>{const button=document.createElement('button');button.textContent=`${i+1}  ${s.title}`;button.className=i===stepIndex?'active':'';button.setAttribute('aria-current',i===stepIndex?'step':'false');button.onclick=()=>{stepIndex=i;render();};$('steps').append(button);});}
 $('step-count').textContent=`Section ${stepIndex+1} of ${review.steps.length}`;
 for(const [prefix,s] of [['current',current],['proposed',proposed]]){$(prefix+'-kind').textContent=s.kind.replace(/([a-z])([A-Z])/g,'$1 $2');$(prefix+'-title').textContent=s.title;$(prefix+'-words').textContent=`${s.text.split(/\s+/).length} words`;rich($(prefix+'-text'),s.text);}
 $('proposed-text').classList.toggle('diff-copy',mode==='diff');
 if(mode==='diff'){wordDiff($('proposed-text'),current.text,proposed.text);$('proposed-title').textContent=proposed.title;}
 $('current-pane').hidden=mode!=='compare';$('panes').classList.toggle('single',mode!=='compare');$('diff-legend').hidden=mode!=='diff';
 document.querySelectorAll('[data-mode]').forEach(b=>b.setAttribute('aria-pressed',String(b.dataset.mode===mode)));
 $('previous').disabled=stepIndex===0;$('next').disabled=stepIndex===review.steps.length-1;
}
function showDecision(record){
 if(!record||record.revision!==review.revision)return;
 $('decision-status').textContent=record.action==='approve'?'Wording approved':'Changes requested';
 $('decision-message').textContent=record.action==='approve'?'This exact proposal is saved as approved. Next: integrate the wording, then check the Foundations chapter in the app. No app source changed from this viewer.':'Your notes are saved with this proposal. The next revision can be reviewed here before approval.';
 $('note').value=record.note||'';
 $('reviewed').checked=record.action==='approve';$('approve').disabled=record.action==='approve';
}
async function decide(action){
 if(busy)return;
 if(action==='request_changes'&&!$('note').value.trim()){$('decision-message').textContent='Add a short note so the next revision addresses what you want changed.';$('note').focus();return;}
 busy=true;$('approve').disabled=true;$('request').disabled=true;
 try{const response=await fetch('/api/decision',{method:'POST',headers:{'Content-Type':'application/json','X-Review-Token':review.token},body:JSON.stringify({action,note:$('note').value,revision:review.revision,sourceHash:review.current.sourceHash})});const data=await response.json();if(!response.ok)throw Error(data.error);showDecision(data);review.decision=data;}
 catch(error){$('decision-message').textContent=error.message;}
 finally{busy=false;$('request').disabled=false;$('approve').disabled=!$('reviewed').checked||review.decision?.action==='approve';}
}
$('previous').onclick=()=>{stepIndex--;render();};$('next').onclick=()=>{stepIndex++;render();};
for(const button of document.querySelectorAll('[data-mode]'))button.onclick=()=>{mode=button.dataset.mode;render();};
$('approve').onclick=()=>decide('approve');$('request').onclick=()=>decide('request_changes');
$('reviewed').onchange=()=>{$('approve').disabled=!$('reviewed').checked||review?.decision?.action==='approve';};
function referenceDialog(){if(review)$('reference').showModal();}
$('reference-button').onclick=referenceDialog;$('accepted-lesson').onclick=referenceDialog;$('close-reference').onclick=()=>$('reference').close();
let size=16;try{size=Math.max(12,Math.min(24,Number(localStorage.getItem('kwip-review-size'))||16));}catch{}
function setSize(delta){size=Math.max(12,Math.min(24,size+delta));document.documentElement.style.setProperty('--copy-size',size+'px');try{localStorage.setItem('kwip-review-size',String(size));}catch{}}
setSize(0);$('smaller').onclick=()=>setSize(-1);$('larger').onclick=()=>setSize(1);
(async()=>{try{const r=await fetch('/api/review');review=await r.json();if(!r.ok)throw Error(review.error);$('title').textContent=review.title;$('lesson-type').textContent=review.type.toUpperCase();$('scope-label').textContent=review.scopeLabel||review.scope;$('check-count').textContent=`${review.sourceChecks.length} checks`;$('review-count').textContent=`I’ve reviewed all ${review.steps.length} sections.`;$('summary').textContent=review.summary;$('review-note').textContent=review.reviewNote;$('source-path').textContent=review.current.sourcePath+' · source '+review.current.sourceHash.slice(0,12);for(const text of review.sourceChecks){const li=document.createElement('li');li.textContent=text;$('source-checks').append(li);}rich($('reference-text'),review.reference);if(review.applied){$('review-stage').textContent='Applied checkpoint';$('proposal-label').textContent='Applied wording';document.querySelector('.decision').hidden=true;}$('workspace').hidden=false;render();showDecision(review.decision);}catch(error){$('load-error').hidden=false;$('load-error').textContent=error.message;$('title').textContent='Review unavailable';}})();
