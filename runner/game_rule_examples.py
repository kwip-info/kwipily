"""Authored fixtures; no game-name dispatch in the evaluator."""
import copy

def integer(lo,hi,initial=None):
    s=dict(type='integer',minimum=lo,maximum=hi)
    if initial is not None:s['initial']=initial
    return s

def base(name):return dict(version=1,name=name,parameters={},state={},actions={},currentPlayer='state.to_move',legal='True',transitions=[],terminal=[],observations=[],reward='0 if not outcome.terminal or outcome.winner == None else (1 if outcome.winner == player else -1)',invariants=[],tests=[])

def take_away():
    x=base('Take-away');x.update(state=dict(remaining=integer(0,21,21),to_move=integer(0,1,0)),actions=dict(take=integer(1,3)),legal='action.take <= state.remaining',transitions=[dict(name='Take objects',when='True',updates=dict(remaining='state.remaining - action.take',to_move='1 - state.to_move'))],terminal=[dict(name='Last object',when='state.remaining == 0',winner='1 - state.to_move')],observations=['state.remaining / 21'])
    x['tests']=[dict(name='Take 3 from 5',state=dict(remaining=5,to_move=0),actions=[dict(take=3)],expect=dict(state=dict(remaining=2,to_move=1),terminal=False)),dict(name='Too many objects',state=dict(remaining=2,to_move=0),actions=[dict(take=3)],expectError='illegal',expect=dict(state=dict(remaining=2,to_move=0))),dict(name='Last object wins',state=dict(remaining=1,to_move=1),actions=[dict(take=1)],expect=dict(terminal=True,winner=1,rewards=[-1,1]))]
    return x

def board_game(rows,cols,connect,gravity):
    x=base('Gravity placement' if gravity else 'Grid placement');count=rows*cols;x['parameters']=dict(rows=rows,cols=cols,connect=connect)
    x['state']=dict(board=dict(type='array',length=count,items=integer(0,2),initial=[0]*count),to_move=integer(0,1,0))
    x['actions']=dict(column=integer(0,cols-1)) if gravity else dict(cell=integer(0,count-1))
    x['legal']='state.board[action.column] == 0' if gravity else 'state.board[action.cell] == 0'
    index='max([r for r in range(params.rows) if state.board[r * params.cols + action.column] == 0]) * params.cols + action.column' if gravity else 'action.cell'
    x['transitions']=[dict(name='Place token',when='True',updates=dict(board=f'set_at(state.board, {index}, state.to_move + 1)',to_move='1 - state.to_move'))]
    # Enumerate finite start cells and direction offsets using generic arithmetic,
    # array indexing, bounded quantification and short-circuit bounds, not a game helper.
    win=[]
    for dr,dc in ((0,1),(1,0),(1,1),(1,-1)):
        win.append(f'any([all([state.board[(r + k * {dr}) * params.cols + c + k * {dc}] == 2 - state.to_move for k in range(params.connect)]) for r in range(params.rows) for c in range(params.cols) if 0 <= r + (params.connect - 1) * {dr} < params.rows and 0 <= c + (params.connect - 1) * {dc} < params.cols])')
    victory='('+' or '.join(win)+')'
    x['terminal']=[dict(name='Completed line',when=victory,winner='1 - state.to_move'),dict(name='Full grid draw',when=f'all([v != 0 for v in state.board]) and not {victory}',winner='None')]
    x['observations']=[f'1 if state.board[{i}] == player + 1 else -1 if state.board[{i}] != 0 else 0' for i in range(count)]
    x['invariants']=['abs(sum([1 if v == 1 else -1 if v == 2 else 0 for v in state.board])) <= 1']
    key='column' if gravity else 'cell';moves=[0,1,0,1,0,1,0] if gravity else [0,3,1,4,2]
    x['tests']=[dict(name='First move and turn',actions=[{key:0}],expect=dict(state=dict(to_move=1),terminal=False)),dict(name='Complete a line',actions=[{key:i} for i in moves],expect=dict(terminal=True,winner=0)),dict(name='Reject invalid action',actions=[{key:cols if gravity else count}],expectError='illegal')]
    return x

def examples():return copy.deepcopy({'take-away':take_away(),'placement':board_game(3,3,3,False),'gravity':board_game(6,7,4,True)})
