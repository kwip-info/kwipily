"""Pinned generic opponents and replayed held-out openings; never training targets."""
import random,copy
if __package__:
 from .self_play_training import play,paired_interval
 from .game_episodes import require,digest
else:
 from self_play_training import play,paired_interval
 from game_episodes import require,digest

class Uniform:
 def __init__(self,n):self.n=n
 def predict(self,x):return [0.]*self.n,0.
 def snapshot_digest(self):return digest(['uniform-zero',self.n])
class RandomLegal(Uniform):pass

def validate_opponents(c,opponents=None):
 names=c.evaluation.get('opponents',['initial'])
 require(type(names) is list and 1<=len(names)<=8 and all(type(x) is str for x in names) and len(set(names))==len(names),'Choose 1…8 distinct evaluation opponents.')
 require(all(name in {'initial','random','uniform-search'} or name in (opponents or {}) for name in names),'Select a verified compatible frozen opponent.')
 return names

def evaluate_suite(definition,model,baseline,c,iteration,check,*,opponents=None):
 if c.compute:
  if __package__:from .self_play_acceleration import CachedRules,CachedPolicy
  else:from self_play_acceleration import CachedRules,CachedPolicy
  definition=CachedRules(definition,c.compute.get('cacheBytes',16777216));model=CachedPolicy(model);baseline=CachedPolicy(baseline)
 suite=c.evaluation;names=validate_opponents(c,opponents);openings=suite.get('openings',[[]]);seed=suite.get('seed',8301)
 require(type(names) is list and 1<=len(names)<=8 and len(set(names))==len(names) and all(type(x) is str for x in names),'Choose 1…8 distinct evaluation opponents.')
 require(type(openings) is list and 1<=len(openings)<=32 and all(type(x) is list and len(x)<=128 for x in openings),'Opening suite exceeds bounds.')
 require(type(seed) is int and 0<=seed<=2147483647,'Invalid independent evaluation seed.')
 n=len(model.predict([0.]*__import__('math').prod(model.sample_shape))[0]);available={'initial':baseline,'random':RandomLegal(n),'uniform-search':Uniform(n),**(opponents or {})}
 require(all(name in available for name in names),'Select a verified compatible frozen opponent.')
 rows=[];summaries=[];before=model.snapshot_digest()
 # Separate suite hash is not included in replay rows; collection never consumes these starts.
 for name in names:
  opponent=available[name];pin=opponent.snapshot_digest()
  for mode in ('policy-only','search-assisted'):
   scores=[];counts=dict(wins=0,draws=0,losses=0,truncated=0,failed=0);seat_scores={0:[],1:[]}
   for pair in range(c.evaluationPairs):
    opening=openings[pair%len(openings)];pair_scores=[];pair_seed=seed+iteration*100000+pair
    for seat in (0,1):
     players=[opponent,opponent];players[seat]=model
     try:
      outcome,_,trace=play(definition,players,c,random.Random(pair_seed),check,search=mode=='search-assisted',opening=opening,policy_modes={1-seat:'random' if name=='random' else 'search' if name=='uniform-search' else mode})
      score=None if outcome['status']!='terminal' else .5 if outcome['winner'] is None else float(outcome['winner']==seat)
      if score is None:counts['truncated' if outcome['status']=='truncated' else 'failed']+=1
      else:counts['wins' if score==1 else 'draws' if score==.5 else 'losses']+=1;pair_scores.append(score);seat_scores[seat].append(score)
      rows.append(dict(iteration=iteration,opponent=name,opponentSHA256=pin,mode=mode,pair=pair,seat=seat,seed=pair_seed,openingSHA256=digest(opening),candidateSimulations=c.search['simulations'] if mode=='search-assisted' else 0,opponentSimulations=c.search['simulations'] if name=='uniform-search' or mode=='search-assisted' and name!='random' else 0,candidateRealizedSimulations=sum(r['simulations'] for r in trace if r['actor']==seat),opponentRealizedSimulations=sum(r['simulations'] for r in trace if r['actor']!=seat),candidateDecisionSeconds=sum(r['decisionSeconds'] for r in trace if r['actor']==seat),opponentDecisionSeconds=sum(r['decisionSeconds'] for r in trace if r['actor']!=seat),status=outcome['status'],winner=outcome['winner'],score=score,error=outcome['error']))
     except (ValueError,TimeoutError) as error:
      counts['failed']+=1;rows.append(dict(iteration=iteration,opponent=name,mode=mode,pair=pair,seat=seat,seed=pair_seed,openingSHA256=digest(opening),status='failed',score=None,error=str(error)))
    if len(pair_scores)==2:scores.append(sum(pair_scores)/2)
   terminal=counts['wins']+counts['draws']+counts['losses']
   summaries.append(dict(opponent=name,opponentSHA256=pin,mode=mode,**counts,terminal=terminal,attempts=c.evaluationPairs*2,completePairs=len(scores),score=(counts['wins']+.5*counts['draws'])/terminal if terminal else None,pairedInterval=paired_interval(scores,seed),seatScores={str(k):sum(v)/len(v) if v else None for k,v in seat_scores.items()}))
  require(pin==opponent.snapshot_digest(),'Evaluation opponent mutated.')
 require(before==model.snapshot_digest(),'Evaluation candidate mutated.')
 return rows,summaries
