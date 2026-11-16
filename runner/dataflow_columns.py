"""Exact column-grid transformations over bounded row streams.

Legacy Select/Name and Cast are deliberately unchanged. Names are exact identifiers,
renames simultaneous, and trimming applies only to explicitly converted text.
"""

def validate_config(config):
    rows=config['rows']
    sources=[row['source'] for row in rows]
    outputs=[row['outputName'] for row in rows if row['included']]
    if any(not name.strip() for row in rows for name in (row['source'],row['outputName'])):
        raise ValueError('Columns names must not be blank')
    if len(set(sources))!=len(sources):raise ValueError('Columns source names must be unique')
    if len(set(outputs))!=len(outputs):raise ValueError('Columns output names must be unique')


def compile_selection(table,config):
    validate_config(config)
    configured={row['source'] for row in config['rows']}
    selections=[]
    for row in config['rows']:
        index=table.index(row['source']) # Even excluded stale names must be repaired.
        if row['included']:selections.append((index,row['outputName'],row['targetType']))
    unconfigured=[i for i,column in enumerate(table.columns) if column['name'] not in configured]
    if unconfigured and config['newColumns']=='error':raise ValueError('Unconfigured input columns; refresh the Columns grid')
    if config['newColumns']=='keep':selections.extend((i,table.columns[i]['name'],'keep') for i in unconfigured)
    if not selections:raise ValueError('Columns must retain at least one column')
    names=[name for _,name,_ in selections]
    if len(set(names))!=len(names):raise ValueError('Columns would produce duplicate output names')
    columns=[]
    for index,name,dtype in selections:
        column={**table.columns[index],'name':name}
        if dtype in ('date','datetime','time') or column['dtype'] in ('date','datetime','time') and dtype!='keep':raise ValueError('Temporal columns can be kept or renamed; use Formula Parse, Format or Extract for explicit temporal conversion.')
        if column['dtype'] in ('audio','tensor') and dtype!='keep':raise ValueError('Media columns can be kept or renamed; use Prepare media for explicit conversion.')
        if dtype=='image' and column['dtype']!='image':raise ValueError('Image type requires an existing image column; scalar decoding is not supported')
        if dtype not in ('keep','image'):
            if column['dtype']=='image':raise ValueError('Image columns cannot be cast to scalar values; use an explicit image adapter')
            column.update(dtype=dtype,nullable=config['castFailure']=='null' or (config['missingPolicy']=='preserve' and column['nullable']))
        columns.append(column)
    return columns,selections


def apply(table,config,check,*,share=False):
    if __package__:from . import dataflow as df
    else:import dataflow as df
    columns,selections=compile_selection(table,config)
    if share and all(dtype in ('keep','image') for _,_,dtype in selections):
        if __package__:from .dataflow_shared import projection_table
        else:from dataflow_shared import projection_table
        shared=projection_table(table,columns,[i for i,_,_ in selections],check)
        if shared is not None:return shared
    def pairs():
        for position,(source,rid) in enumerate(df.row_pairs(table)):
            if position%128==0:check()
            output=[]
            for index,name,dtype in selections:
                value=source[index]
                if dtype not in ('keep','image'):
                    if value is None:
                        if config['missingPolicy']=='error':raise ValueError(f'Columns row {position+1}, {name}: missing value requires preserve policy')
                    else:
                        if config['trimText'] and isinstance(value,str):value=value.strip()
                        try:value=df.cast_value(value,dtype)
                        except ValueError as error:
                            if config['castFailure']=='null':value=None
                            else:raise ValueError(f'Columns row {position+1}, {name}: {error}') from error
                output.append(value)
            yield output,rid
    return df.streaming_rows(table,pairs(),columns,check)
