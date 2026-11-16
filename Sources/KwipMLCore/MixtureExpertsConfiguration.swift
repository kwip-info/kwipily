import Foundation

/// A routed operation owns scoped pointwise graph sources, not flattened eager branches.
public struct MixtureExpertsConfiguration:Equatable {
    public struct Expert:Equatable {public var id:String;public var name:String;public var trainable:Bool;public var graph:GraphDocument}
    public static let allowed:Set<String>=["input","output","linear","relu","tanh","sigmoid","gelu","silu","leaky_relu","elu","layer_norm","rms_norm","dropout","add"]
    private final class PresentationCache {
        let lock=NSLock();var values:[String:JSONValue]=[:]
        func stripped(_ text:String,router:Bool)->JSONValue? {
            let key=(router ? "router:":"expert:")+text
            lock.lock();defer{lock.unlock()}
            if let value=values[key]{return value}
            guard let data=text.data(using:.utf8),case .object(var row)=try? JSONDecoder().decode(JSONValue.self,from:data) else{return nil}
            func graph(_ source:[String:JSONValue])->JSONValue {
                var result=source
                if case .array(let nodes)=source["nodes"] {result["nodes"] = .array(nodes.map{item in guard case .object(var node)=item else{return item};for field in ["name","x","y"]{node.removeValue(forKey:field)};return .object(node)})}
                if case .array(let edges)=source["edges"] {result["edges"] = .array(edges.map{item in guard case .object(var edge)=item else{return item};edge.removeValue(forKey:"id");return .object(edge)})}
                return .object(result)
            }
            let value:JSONValue
            if router {value=graph(row)}else{row.removeValue(forKey:"name");if case .object(let source)=row["graph"]{row["graph"]=graph(source)};value = .object(row)}
            if values.count>=64{values.removeAll(keepingCapacity:true)};values[key]=value;return value
        }
    }
    private static let presentationCache=PresentationCache()
    public static func scientificValues(_ values:[String:JSONValue])->[String:JSONValue] {
        var result=values
        for key in ["router","expert_1","expert_2","expert_3","expert_4"] {if let text=values[key]?.stringValue,text.utf8.count<=16384,let value=presentationCache.stripped(text,router:key=="router"){result[key]=value}}
        return result
    }
    public let values:[String:JSONValue]
    public let experts:[Expert]
    public let router:GraphDocument
    public let parameterCount:Int
    public let trainableCount:Int
    public let inputFeatures:Int
    public let outputFeatures:Int
    public let topK:Int
    public init(values:[String:JSONValue],registry:ComponentRegistry)throws {
        guard let definition=registry.component(type:"mixture_experts"),Set(values.keys)==Set(definition.fields.map(\.key)),definition.fields.allSatisfy({values[$0.key].map($0.accepts)==true}),values["version"]==1,
              let count=values["expert_count"]?.integerValue,let k=values["top_k"]?.integerValue,k<=count,
              let width=values["input_features"]?.integerValue,let output=values["output_features"]?.integerValue else {throw GraphConfigurationError("Use the declared MoE fields; experts per unit must not exceed the2–4 expert count.")}
        self.values=values;inputFeatures=width;outputFeatures=output;topK=k
        func object(_ text:String)throws->[String:JSONValue] {
            guard text.unicodeScalars.count<=4096,let data=text.data(using:.utf8),case .object(let value)=try JSONDecoder().decode(JSONValue.self,from:data) else {throw GraphConfigurationError("Use bounded inert JSON for expert and router graphs.")};return value
        }
        func graph(_ value:JSONValue,output:Int,qualified:Bool=true)throws->(GraphDocument,Int) {
            guard case .object(let row)=value,Set(row.keys)==["schemaVersion","nodes","edges"],row["schemaVersion"]==2,
                  case .array(let ns)=row["nodes"],(2...12).contains(ns.count),case .array(let es)=row["edges"],es.count<=24 else {throw GraphConfigurationError("Pointwise graph requires schema2,2–12 nodes and at most24 edges.")}
            var nodes:[GraphNode]=[],edges:[GraphEdge]=[],parameters=0
            for item in ns {
                guard case .object(let n)=item,Set(n.keys).isSubset(of:["id","type","name","x","y","config"]),let id=n["id"]?.stringValue,let type=n["type"]?.stringValue,Self.allowed.contains(type),case .object(let c)=n["config"] else {throw GraphConfigurationError("Only independent pointwise graph operations are available inside an expert/router.")}
                nodes.append(.init(id:id,type:type,name:n["name"]?.stringValue ?? type,x:n["x"]?.doubleValue ?? Double(nodes.count)*180,y:n["y"]?.doubleValue ?? 100,config:c))
                if type=="linear",let a=c["in_features"]?.integerValue,let b=c["out_features"]?.integerValue {parameters+=a*b+(c["bias"]==true ? b:0)}
                if ["layer_norm","rms_norm"].contains(type),case .array(let dims)=c["normalized_shape"],c["elementwise_affine"]==true {parameters+=dims.compactMap(\.integerValue).reduce(1,*)*(type=="layer_norm" && c["bias"]==true ? 2:1)}
            }
            for item in es {guard case .object(let e)=item,Set(e.keys)==["id","source","sourcePort","target","targetPort"],let id=e["id"]?.stringValue,let source=e["source"]?.stringValue,let sp=e["sourcePort"]?.stringValue,let target=e["target"]?.stringValue,let tp=e["targetPort"]?.stringValue else {throw GraphConfigurationError("Pointwise connection requires exact named endpoints.")};edges.append(.init(id:id,source:source,sourcePort:sp,target:target,targetPort:tp))}
            let document=GraphDocument(name:"Pointwise graph",nodes:nodes,edges:edges)
            if !qualified{return(document,parameters)}
            guard nodes.filter({$0.type=="input"}).count==1,let source=nodes.first(where:{$0.type=="input"}),source.config==["features":.integer(Int64(width))] else {throw GraphConfigurationError("Expert/router input must match the MoE feature width.")}
            let validation=GraphValidator.validate(document,registry:registry,context:.explicitObjective)
            guard validation.isValid else {throw GraphConfigurationError(validation.issues.map(\.message).joined(separator:" "))}
            guard let end=nodes.first(where:{$0.type=="output"}),validation.outputTensorContracts[.init(end.id,"out")]?.layout.sample.dimensions == [output] else {throw GraphConfigurationError("Expert output width must match MoE output; router width must match expert count.")}
            return(document,parameters)
        }
        let routerObject=try object(values["router"]!.stringValue!),r=try graph(.object(routerObject),output:count)
        router=r.0;var total=r.1,trainable=r.1,items:[Expert]=[],ids:Set<String>=[]
        for i in 1...4 {
            let row=try object(values["expert_\(i)"]!.stringValue!)
            guard Set(row.keys)==["id","name","trainable","graph"],let id=row["id"]?.stringValue,id.range(of:"^[A-Za-z][A-Za-z0-9_]{0,31}$",options:.regularExpression) != nil,ids.insert(id).inserted,let name=row["name"]?.stringValue,(1...80).contains(name.count),let trained=row["trainable"]?.boolValue,let source=row["graph"] else {throw GraphConfigurationError("Expert requires a unique stable ID, bounded name, trainability and pointwise graph.")}
            let checked=try graph(source,output:output,qualified:i<=count)
            if i<=count {items.append(.init(id:id,name:name,trainable:trained,graph:checked.0));total+=checked.1;trainable+=trained ? checked.1:0}
        }
        guard total<=2_000_000 else {throw GraphConfigurationError("All resident experts and router exceed the2million parameter limit.")}
        experts=items;parameterCount=total;trainableCount=trainable
    }
    public func outputLayout(_ input:BatchedTensorLayout)throws->BatchedTensorLayout {
        guard (2...3).contains(input.rank),input.sample.dimensions.last==inputFeatures else {throw GraphConfigurationError("MoE needs [N,D] or [N,T,D] with the declared final feature width.")}
        return try .init(sample:Array(input.sample.dimensions.dropLast())+[outputFeatures])
    }
    public static func graphValue(_ graph:GraphDocument)->JSONValue {
        .object(["schemaVersion":2,"nodes":.array(graph.nodes.map{.object(["id":.string($0.id),"name":.string($0.name),"type":.string($0.type),"x":.number($0.x),"y":.number($0.y),"config":.object($0.config)])}),"edges":.array(graph.edges.map{.object(["id":.string($0.id),"source":.string($0.source),"sourcePort":.string($0.sourcePort),"target":.string($0.target),"targetPort":.string($0.targetPort)])})])
    }
    public static func pointwiseGraph(_ value:JSONValue)throws->GraphDocument {
        guard case .object(var row)=value,case .array(let nodes)=row["nodes"] else {throw GraphConfigurationError("Pointwise graph source is missing.")}
        row["name"] = .string("Pointwise graph")
        row["nodes"] = .array(nodes.enumerated().map{index,item in
            guard case .object(var node)=item else{return item}
            if node["name"]==nil{node["name"] = node["type"]}
            if node["x"]==nil{node["x"] = .number(Double(index)*180)}
            if node["y"]==nil{node["y"] = 100}
            return .object(node)
        })
        return try JSONDecoder().decode(GraphDocument.self,from:JSONEncoder().encode(JSONValue.object(row)))
    }
}
