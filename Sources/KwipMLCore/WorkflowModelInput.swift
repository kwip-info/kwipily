import Foundation

/// Model-owned authoring settings backed by visible executable tensor boundaries.
/// Legacy ownership is inferred for presentation only; opening never rewrites a project.
public enum WorkflowModelInput {
    public static let componentIDs:Set<String> = ["input","index_input"]
    public static func isBoundary(_ node:WorkflowNode)->Bool {
        if case .modelPrimitive(let type,_,_)=node.payload {return componentIDs.contains(type)}
        return false
    }
    public static func ancestors(of definition:WorkflowNodeAddress,in document:WorkflowDocument)->Set<WorkflowNodeAddress> {
        let incoming=Dictionary(grouping:document.edges,by:{$0.to.address})
        var pending=[definition],seen=Set<WorkflowNodeAddress>()
        while let address=pending.popLast() {
            guard seen.insert(address).inserted else {continue}
            pending += (incoming[address] ?? []).map {$0.from.address}
        }
        return seen
    }
    public static func boundary(for definition:WorkflowNodeAddress,in document:WorkflowDocument,name:String?=nil)throws->WorkflowNode? {
        guard let node=document.nodes.first(where:{$0.address==definition}),WorkflowModelBoundary.isDefinition(node) else {
            throw DataFlowContractError("Select a Model definition to configure its input.")
        }
        if let named=document.namedModelInputs?[definition.localID],!named.isEmpty {
            let selected=name ?? named.keys.sorted().first!
            guard let id=named[selected] else {return nil}
            return document.nodes.first {$0.address == .init(domain:.model,localID:id)}
        }
        let region=ancestors(of:definition,in:document)
        let upstream=document.nodes.filter {node in
            guard region.contains(node.address),case .modelPrimitive(let type,_,_)=node.payload else {return false}
            return componentIDs.contains(type) || type=="bag_input"
        }
        let owned=document.modelInputs?[definition.localID].flatMap {id in document.nodes.first {$0.address == .init(domain:.model,localID:id)}}
        let boundaries=Set(upstream.map(\.address)).union(owned.map {[$0.address]} ?? [])
        guard boundaries.count<=1 else {throw DataFlowContractError("This model has more than one input boundary. Repair its entry connections before editing input settings.")}
        return owned ?? upstream.first
    }
    public static func destinations(of boundary:WorkflowNode,in document:WorkflowDocument)->[WorkflowEndpoint] {
        document.edges.filter {$0.from.address==boundary.address && $0.from.port=="out" && $0.kind == .tensor}.map(\.to)
    }
    static func migrateOwnershipMetadata(_ document:inout WorkflowDocument) {
        if document.schemaVersion<14 {
            let version=document.schemaVersion>=11 ? 3 : 2
            var sources=document.modelSourceVersions ?? [:]
            for node in document.nodes {
                if WorkflowModelBoundary.isDefinition(node) {sources[node.address.localID]=version}
            }
            document.modelSourceVersions=sources
            document.schemaVersion=14
        }
    }
    public static func validateOwnership(_ document:WorkflowDocument)throws {
        if let models=document.namedModelInputs {
            guard document.schemaVersion>=19,models.count<=128 else {throw DataFlowContractError("Named model inputs require workflow schema19.")}
            for (modelID,names) in models {
                guard document.nodes.contains(where:{$0.address.localID==modelID && WorkflowModelBoundary.isDefinition($0)}),(1...8).contains(names.count),Set(names.values).count==names.count else {throw DataFlowContractError("Named input ownership is invalid.")}
                for (name,id) in names {guard name.range(of:"^[A-Za-z][A-Za-z0-9_]{0,63}$",options:.regularExpression) != nil,document.nodes.contains(where:{$0.address == .init(domain:.model,localID:id) && isBoundary($0)}) else {throw DataFlowContractError("Named input must reference an actual numeric/index boundary.")}}
            }
        }
        if let versions=document.modelSourceVersions {
            guard document.schemaVersion>=14,versions.count<=128 else {throw DataFlowContractError("Model source version retention requires workflow schema14.")}
            for (id,version) in versions {
                guard [2,3].contains(version),let node=document.nodes.first(where:{$0.address.localID==id && WorkflowModelBoundary.isDefinition($0)}),WorkflowModelBoundary.isDefinition(node) else {throw DataFlowContractError("Retained model source versions must name existing Model definitions and supported versions.")}
            }
        }
        guard let owners=document.modelInputs else {return}
        guard document.schemaVersion>=14,owners.count<=128 else {throw DataFlowContractError("Model input ownership requires workflow schema14.")}
        for (modelID,inputID) in owners {
            guard let model=document.nodes.first(where:{$0.address.localID==modelID && WorkflowModelBoundary.isDefinition($0)}),WorkflowModelBoundary.isDefinition(model),
                  let input=document.nodes.first(where:{$0.address == .init(domain:.model,localID:inputID)}),isBoundary(input) else {
                throw DataFlowContractError("Model input ownership must reference a Model definition and an existing scalar tensor input boundary.")
            }
        }
    }
}
