"""Version-one deterministic sampling without replacement for bounded typed tables.

Rank observation identities by a seeded digest, then retain original row order.
Increasing the count with the same seed yields nested membership. This reduces
subsequent work; it does not change the source reader's materialization budget.
"""
import hashlib
import heapq


def sample_rows(table, count, seed):
    if __package__:
        from .dataflow import MAX_ROWS, derive, StoredIDs, EAGER_CELL_LIMIT
    else:
        from dataflow import MAX_ROWS, derive, StoredIDs, EAGER_CELL_LIMIT
    if type(count) is not int or not 1 <= count <= min(MAX_ROWS, len(table.rows)):
        raise ValueError('Row sample count must be an integer between 1 and the available row count, up to 100000.')
    if type(seed) is not int or not 0 <= seed <= 4294967295:
        raise ValueError('Row sample seed must be an integer between 0 and 4294967295.')
    if count*len(table.columns)>EAGER_CELL_LIMIT:raise ValueError('Row Sample exceeds its in-memory cell budget; choose fewer rows or columns.')
    ids=table.row_ids
    if len(ids)!=len(table.rows) or any(not isinstance(identity,str) or not identity for identity in ids) or (not isinstance(ids,StoredIDs) and len(set(ids))!=len(ids)):
        raise ValueError('Row sampling requires unique nonempty observation identities.')
    prefix=b'kwip.sample.rows.v1\n'+str(seed).encode('ascii')+b'\n'
    def rank(index):
        identity=ids[index].encode('utf-8')
        return (hashlib.sha256(prefix+str(len(identity)).encode('ascii')+b':'+identity).digest(),identity)
    if isinstance(ids,StoredIDs):
        def ranked():
            for index,identity_text in enumerate(ids):
                identity=identity_text.encode('utf-8')
                yield (hashlib.sha256(prefix+str(len(identity)).encode('ascii')+b':'+identity).digest(),identity,index)
        retained=sorted(item[2] for item in heapq.nsmallest(count,ranked()))
    else:
        chosen=set(sorted(range(len(ids)),key=rank)[:count])
        retained=[i for i in range(len(ids)) if i in chosen]
    return derive(table,rows=[table.rows[i] for i in retained],ids=[ids[i] for i in retained])
