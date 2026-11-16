"""Bounded metadata pages over a pinned pack; no population-sized JSON index.

Each record covers a positive ordered span. Builders retain at most128 entries
per level. A scoped lookup validates its path; records() traverses all metadata.
Consumers still validate their own table/column/identity geometry.
"""
FANOUT=128
MAX_INTEGER=9_007_199_254_740_991
MAX_HEIGHT=8


class IndexBuilder:
    """Incremental ordered pages; at most FANOUT entries at each level."""
    def __init__(self,pack):self.pack=pack;self.levels=[];self.leaf=[];self.total=0;self.closed=False
    def _node(self,height,entries):
        count=sum(entry['count'] for entry in entries)
        if height>MAX_HEIGHT or count>MAX_INTEGER:raise ValueError('Packed index exceeds its protocol integer/depth boundary.')
        reference=self.pack.add_json({'schemaVersion':1,'kind':'orderedIndex','height':height,'count':count,'entries':entries})
        return {'reference':reference,'count':count,'height':height}
    def _push(self,level,entry):
        while len(self.levels)<=level:self.levels.append([])
        self.levels[level].append(entry)
        if len(self.levels[level])==FANOUT:
            entries=self.levels[level];self.levels[level]=[]
            merged=self._node(level+1,[{'reference':item['reference'],'count':item['count']} for item in entries])
            self._push(level+1,merged)
    def append(self,count,value):
        if self.closed:raise ValueError('Packed index builder is closed.')
        if type(count) is not int or not 1<=count<=MAX_INTEGER or self.total+count>MAX_INTEGER:raise ValueError('Packed index spans must be positive exact protocol integers.')
        self.total+=count;self.leaf.append({'count':count,'value':value})
        if len(self.leaf)==FANOUT:self._push(0,self._node(0,self.leaf));self.leaf=[]
    def finish(self):
        if self.closed:raise ValueError('Packed index builder is closed.')
        self.closed=True
        if self.leaf:self._push(0,self._node(0,self.leaf));self.leaf=[]
        if not self.levels:return self._node(0,[])
        level=0
        while True:
            entries=self.levels[level]
            if entries:
                if len(entries)==1 and not any(self.levels[level+1:]):return entries[0]
                self.levels[level]=[]
                self._push(level+1,self._node(level+1,[{'reference':item['reference'],'count':item['count']} for item in entries]))
            level+=1


def build_index(pack,records):
    builder=IndexBuilder(pack)
    for count,value in records:builder.append(count,value)
    return builder.finish()


class OrderedIndex:
    def __init__(self,reader,root,*,maximum_count):
        if (type(maximum_count) is not int or not 0<=maximum_count<=MAX_INTEGER
            or type(root) is not dict or set(root)!={'reference','count','height'}
            or type(root['reference']) is not dict
            or type(root['count']) is not int or not 0<=root['count']<=maximum_count
            or type(root['height']) is not int or not 0<=root['height']<=MAX_HEIGHT):
            raise ValueError('Invalid bounded packed index root.')
        self.reader=reader;self._reference=dict(root['reference']);self.count=root['count'];self.height=root['height'];self.maximum=maximum_count
        self._node(reader,self._reference,self.height,self.count)
    def _node(self,reader,reference,height,count):
        value,children=reader.read_node(reference) if hasattr(reader,'read_node') else (reader.read_json(reference),reader)
        if (type(value) is not dict or set(value)!={'schemaVersion','kind','height','count','entries'}
            or type(value['schemaVersion']) is not int or value['schemaVersion']!=1 or value['kind']!='orderedIndex'
            or type(value['height']) is not int or value['height']!=height
            or type(value['count']) is not int or value['count']!=count
            or type(value['entries']) is not list or len(value['entries'])>FANOUT
            or (not value['entries'])!=(count==0) or count==0 and height!=0):
            raise ValueError('Packed index geometry differs from its parent.')
        field='value' if height==0 else 'reference';total=0
        for entry in value['entries']:
            if (type(entry) is not dict or set(entry)!={'count',field} or type(entry['count']) is not int
                or not 1<=entry['count']<=self.maximum
                or field=='reference' and type(entry[field]) is not dict):raise ValueError('Invalid packed index entry.')
            total+=entry['count']
        if total!=count:raise ValueError('Packed index span count differs.')
        return value['entries'],children
    def locate(self,position):
        value,offset,_=self.locate_node(position)
        return value,offset
    def locate_node(self,position):
        value,offset,owner,_=self.locate_span(position)
        return value,offset,owner
    def locate_span(self,position):
        if type(position) is not int or not 0<=position<self.count:raise IndexError('Packed index position is outside its population.')
        reference=self._reference;height=self.height;count=self.count;reader=self.reader
        while True:
            entries,children=self._node(reader,reference,height,count)
            for entry in entries:
                if position>=entry['count']:position-=entry['count'];continue
                if height==0:return entry['value'],position,children,entry['count']
                reference=entry['reference'];height-=1;count=entry['count'];reader=children;break
            else:raise ValueError('Packed index has no matching span.')
    def records(self):
        def visit(reader,reference,height,count):
            entries,children=self._node(reader,reference,height,count)
            for entry in entries:
                if height==0:yield entry['count'],entry['value']
                else:yield from visit(children,entry['reference'],height-1,entry['count'])
        yield from visit(self.reader,self._reference,self.height,self.count)
