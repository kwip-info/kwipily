import Foundation

/// One main-canvas container over the existing routed source. Qualified child IDs are
/// presentation/command addresses only; worker graph IDs and parameter keys never change.
public enum WorkflowMixtureContainer {
    public struct Chain {
        public let id:String
        public let key:String
        public let name:String
        public let graph:GraphDocument
    }
    public static func contains(_ document:WorkflowDocument,root:WorkflowNodeAddress)->Bool {
        guard let node=document.nodes.first(where:{$0.address==root}),case .modelPrimitive("mixture_experts",_,_)=node.payload else{return false};return true
    }
    private struct PresentationSource {
        let config:[String:JSONValue]
        let chains:[Chain]
        let definition:CompositeDefinition
    }
    private final class PresentationCache {
        let lock=NSLock()
        var entries:[PresentationSource]=[]
        func source(_ config:[String:JSONValue])throws->PresentationSource {
            lock.lock();defer{lock.unlock()}
            if let cached=entries.first(where:{$0.config==config}){return cached}
            let chains=try WorkflowMixtureContainer.decodeChains(config)
            let result=PresentationSource(config:config,chains:chains,definition:WorkflowMixtureContainer.makeDefinition(chains))
            if entries.count>=8{entries.removeFirst()};entries.append(result)
            return result
        }
    }
    private static let presentationCache=PresentationCache()
    private static func source(in document:WorkflowDocument,root:WorkflowNodeAddress)throws->PresentationSource {
        guard let node=document.nodes.first(where:{$0.address==root}),case .modelPrimitive("mixture_experts",_,let config)=node.payload else{throw failure("Choose a current MoE container.")}
        return try presentationCache.source(config)
    }
    public static func chains(in document:WorkflowDocument,root:WorkflowNodeAddress)throws->[Chain] {try source(in:document,root:root).chains}
    private static func decodeChains(_ config:[String:JSONValue])throws->[Chain] {
        guard let count=config["expert_count"]?.integerValue,(2...4).contains(count) else{throw failure("Choose a current MoE container.")}
        func value(_ key:String)throws->JSONValue {
            guard let text=config[key]?.stringValue,text.unicodeScalars.count<=4096 else{throw failure("MoE chain source exceeds its bounds.")}
            return try JSONDecoder().decode(JSONValue.self,from:Data(text.utf8))
        }
        var result=[Chain(id:"$router",key:"router",name:"Router",graph:try MixtureExpertsConfiguration.pointwiseGraph(value("router")))]
        for index in 1...count {
            guard case .object(let row)=try value("expert_\(index)"),let id=row["id"]?.stringValue,let name=row["name"]?.stringValue,let graph=row["graph"] else{throw failure("Expert chain identity is missing.")}
            result.append(.init(id:id,key:"expert_\(index)",name:name,graph:try MixtureExpertsConfiguration.pointwiseGraph(graph)))
        }
        guard Set(result.map(\.id)).count==result.count else{throw failure("Expert identities must be unique.")}
        return result
    }
    public static func qualified(_ chain:String,_ local:String)->String {chain+"/"+local}
    public static func split(_ qualified:String)throws->(chain:String,local:String) {
        guard let slash=qualified.firstIndex(of:"/"),slash != qualified.startIndex else{throw failure("Select an expert chain inside this MoE container.")}
        let local=String(qualified[qualified.index(after:slash)...]);guard !local.isEmpty else{throw failure("Chain member identity is missing.")}
        return(String(qualified[..<slash]),local)
    }
    /// A display-only facade for the shared inline canvas. Never inserted in definitions
    /// or lowered as an ordinary eager composite.
    public static func definition(in document:WorkflowDocument,root:WorkflowNodeAddress)throws->CompositeDefinition {try source(in:document,root:root).definition}
    private static func makeDefinition(_ chains:[Chain])->CompositeDefinition {
        var nodes:[GraphNode]=[],edges:[GraphEdge]=[]
        for (row,chain) in chains.filter({$0.key != "router"}).enumerated() {
            let interior=Set(chain.graph.nodes.filter{!["input","output"].contains($0.type)}.map(\.id))
            for (column,var node) in chain.graph.nodes.filter({interior.contains($0.id)}).enumerated() {
                node.id=qualified(chain.id,node.id);node.x=Double(column)*325;node.y=Double(row)*225
                nodes.append(node)
            }
            edges += chain.graph.edges.filter{interior.contains($0.source) && interior.contains($0.target)}.map{.init(id:qualified(chain.id,$0.id),source:qualified(chain.id,$0.source),sourcePort:$0.sourcePort,target:qualified(chain.id,$0.target),targetPort:$0.targetPort)}
        }
        let targets=chains.filter{$0.key != "router"}.flatMap {chain -> [CompositePortReference] in
            let entries=Set(chain.graph.nodes.filter{$0.type=="input"}.map(\.id))
            let interior=Set(chain.graph.nodes.filter{!["input","output"].contains($0.type)}.map(\.id))
            return chain.graph.edges.filter{entries.contains($0.source) && interior.contains($0.target)}.map{.init(qualified(chain.id,$0.target),$0.targetPort)}
        }
        return .init(reference:.init("moe-display"),title:"Mixture of experts",body:.init(nodes:nodes,edges:edges),inputs:targets.isEmpty ? []:[.init("in",targets:targets)],outputs:[])
    }
    public static func apply(_ command:WorkflowInlineContainer.Command,to document:WorkflowDocument,root:WorkflowNodeAddress,registry:ComponentRegistry,makeID:()->String)throws->WorkflowDocument {
        try WorkflowCodec.validate(document)
        guard document.drafts.isEmpty,document.revision<UInt64.max,let index=document.nodes.firstIndex(where:{$0.address==root}),case .modelPrimitive(let type,let version,var config)=document.nodes[index].payload else{throw failure("Resolve pending drafts before editing this MoE container.")}
        var sources=try chains(in:document,root:root),changed=Set<Int>()
        func edit(_ id:String,_ action:(inout GraphDocument,String)throws->Void)throws {
            let address=try split(id)
            guard let i=sources.firstIndex(where:{$0.id==address.chain}) else{throw failure("The expert chain no longer exists.")}
            var graph=sources[i].graph;try action(&graph,address.local)
            sources[i] = .init(id:sources[i].id,key:sources[i].key,name:sources[i].name,graph:graph);changed.insert(i)
        }
        func graphCommand(_ id:String,_ build:(String)->GraphCommand)throws {
            try edit(id){graph,local in graph=try GraphCommandEngine.apply(build(local),to:.init(document:graph),registry:registry,makeID:makeID).after.document}
        }
        func connect(_ from:CompositePortReference,_ to:CompositePortReference)throws {
            let a=try split(from.componentID),b=try split(to.componentID)
            guard a.chain==b.chain else{throw failure("Each expert has its own Input → Output chain. Cross-expert wires would change routed execution and are not allowed.")}
            try graphCommand(from.componentID){local in .connect(source:.init(local,from.portID),target:.init(b.local,to.portID))}
        }
        func perform(_ command:WorkflowInlineContainer.Command,depth:Int=0)throws {
            guard depth<=8 else{throw failure("Too many nested MoE commands.")}
            switch command {
            case .batch(let commands):guard commands.count<=128 else{throw failure("Too many MoE commands.")};for command in commands{try perform(command,depth:depth+1)}
            case .configure(let id,let values):try edit(id){graph,local in
                guard let i=graph.nodes.firstIndex(where:{$0.id==local}) else{throw failure("Choose a current chain member.")}
                var node=graph.nodes[i];node.config=values;_=try GraphMigration.effectiveConfiguration(of:node,schemaVersion:2,registry:registry);graph.nodes[i]=node
            }
            case .rename(let id,let name):try edit(id){graph,local in
                guard name.count<=500,let i=graph.nodes.firstIndex(where:{$0.id==local}) else{throw failure("Choose a current member and bounded label.")};graph.nodes[i].name=name
            }
            case .setActivation(let id,let function,let values):try graphCommand(id){.setActivation(nodeID:$0,function:function,configuration:values)}
            case .setLinear(let id,let variant,let values):try graphCommand(id){.setLinear(nodeID:$0,variant:variant,configuration:values)}
            case .setNormalization(let id,let variant,let values):try graphCommand(id){.setNormalization(nodeID:$0,variant:variant,configuration:values)}
            case .insertNode(let node):try edit(node.id){graph,local in
                guard !graph.nodes.contains(where:{$0.id==local}),!["input","output"].contains(node.type),MixtureExpertsConfiguration.allowed.contains(node.type) else{throw failure("Add an independent pointwise layer to an existing expert chain; its Input and Output remain.")}
                var node=node;node.id=local;_=try GraphMigration.effectiveConfiguration(of:node,schemaVersion:2,registry:registry);graph.nodes.append(node)
            }
            case .delete(let id):try edit(id){graph,local in
                guard let node=graph.nodes.first(where:{$0.id==local}),!["input","output"].contains(node.type) else{throw failure("Keep the chain's Input and Output. Remove an expert through the MoE configuration.")}
                graph=try GraphCommandEngine.apply(.delete(.init(nodeIDs:[local])),to:.init(document:graph),registry:registry).after.document
            }
            case .connect(let from,let to):try connect(from,to)
            case .connectWorkflow(let from,let to):
                let definition=try self.definition(in:document,root:root)
                guard let a=definition.body.nodes.first(where:{WorkflowInlineContainer.address(root:root,childID:$0.id)==from.address}),let b=definition.body.nodes.first(where:{WorkflowInlineContainer.address(root:root,childID:$0.id)==to.address}) else{throw failure("Connect within one expert chain. External wires use the MoE container's ports.")}
                try connect(.init(a.id,from.port),.init(b.id,to.port))
            case .disconnect(let id):try edit(id){graph,local in
                guard graph.edges.contains(where:{$0.id==local}) else{throw failure("The selected chain wire no longer exists.")};graph.edges.removeAll{$0.id==local}
            }
            default:throw failure("Use pointwise chain edits inside MoE. Per-layer recipes, nested containers and moving layers across its boundary are unavailable.")
            }
        }
        try perform(command)
        let encoder=JSONEncoder();encoder.outputFormatting=[.sortedKeys]
        for i in changed {
            let chain=sources[i],graph=chain.graph
            guard (2...12).contains(graph.nodes.count),graph.edges.count<=24,graph.nodes.filter({$0.type=="input"}).count==1,graph.nodes.filter({$0.type=="output"}).count==1,graph.nodes.allSatisfy({MixtureExpertsConfiguration.allowed.contains($0.type) && ($0.parameterSettings?.isEmpty ?? true)}) else{throw failure("Each expert/router keeps one Input, one Output and at most12 pointwise nodes/24 wires.")}
            let value:JSONValue
            if chain.key=="router"{value=MixtureExpertsConfiguration.graphValue(graph)}else{
                var row=try JSONDecoder().decode([String:JSONValue].self,from:Data(config[chain.key]!.stringValue!.utf8));row["graph"]=MixtureExpertsConfiguration.graphValue(graph);value = .object(row)
            }
            let text=String(data:try encoder.encode(value),encoding:.utf8)!
            guard text.unicodeScalars.count<=4096 else{throw failure("This expert chain exceeds its4096-character saved-source bound.")}
            config[chain.key] = .string(text)
        }
        var result=document
        result.nodes[index].payload = .modelPrimitive(type:type,configurationVersion:version,config:config)
        result.revision+=1;try WorkflowCodec.validate(result)
        return result
    }
    private static func failure(_ message:String)->DataFlowContractError {.init(message)}
}
