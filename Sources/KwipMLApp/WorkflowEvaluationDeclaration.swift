import Foundation
import KwipMLCore

/// Pure consistency check between a current declaration and the preparation graph
/// actually archived with a frozen model. No files, source bindings or rows are read.
enum WorkflowEvaluationDeclaration {
    static func validate(document:WorkflowDocument,plan:WorkflowEvaluationPlan,archivedDataflow:DataFlowDocument)throws {
        try WorkflowCodec.validate(document)
        try DataFlowCodec.validate(archivedDataflow)
        guard document.projectID == plan.projectID,document.revision == plan.revision,
              plan.membershipValidation == .exactArchivedProtectedPopulation,
              plan.testLineage.role == plan.settings.partitionRole,
              plan.canonicalTestEndpoint == DataFlowEndpoint(node:plan.testLineage.split.localID,port:plan.settings.partitionRole),
              let terminal=document.nodes.first(where:{$0.address == plan.evaluate}),
              case .evaluateTest(let settings)=terminal.payload,
              try WorkflowEvaluationSettings.decode(settings) == plan.settings,
              try WorkflowTrainedModelSource.resolve(in:document,consumer:plan.evaluate)==plan.modelSource else {
            throw failure("Evaluation declaration changed after capture; capture it again.")
        }
        let testEdges=document.edges.filter {$0.to.address == plan.evaluate && $0.to.port == "test"}
        let modelEdges=document.edges.filter {$0.to.address == plan.evaluate && $0.to.port == "model"}
        guard testEdges.count == 1,modelEdges.count == 1,
              testEdges[0].from.address == WorkflowNodeAddress(domain:.data,localID:plan.testEndpoint.node),
              testEdges[0].from.port == plan.testEndpoint.port,
              [.table,.dataset].contains(testEdges[0].kind),
              modelEdges[0].from == WorkflowEndpoint(address:plan.frozenModel,port:"model"),
              modelEdges[0].kind == .frozenModel else {
            throw failure("Evaluation inputs changed after capture; select the frozen model and archived test port again.")
        }
        let currentNodes=document.nodes.compactMap {node->DataFlowNode? in
            guard case .dataOperator(let id,let config)=node.payload else {return nil}
            return .init(id:node.address.localID,operatorID:id,config:config)
        }
        let currentEdges=document.edges.filter {$0.from.address.domain == .data && $0.to.address.domain == .data}.map {
            DataFlowEdge(from:.init(node:$0.from.address.localID,port:$0.from.port),to:.init(node:$0.to.address.localID,port:$0.to.port))
        }
        let current=DataFlowDocument(nodes:currentNodes,edges:currentEdges)
        try DataFlowCodec.validate(current)
        let declared=try region(current,selected:plan.testEndpoint.node)
        let archived=try region(archivedDataflow,selected:plan.testEndpoint.node)
        guard declared.nodes == archived.nodes,declared.edges == archived.edges else {
            throw failure("The selected test source, split, transform or dependency differs from the frozen model's archived preparation. Select a matching archived port; current sources will not be rerun.")
        }
        guard let split=declared.nodes[plan.canonicalTestEndpoint.node],
              ["kwip.data.groupSplit","kwip.data.temporalSplit","kwip.data.partition"].contains(split.operatorID),
              declared.nodes[plan.testLineage.source.localID] != nil,
              plan.testLineage.states.allSatisfy({declared.nodes[$0.localID] != nil}) else {
            throw failure("The protected split or fitted-state declaration no longer matches the captured evaluation plan.")
        }
        // This establishes declaration equality only. The snapshot must also find
        // this exact endpoint's table in the verified artifact inventory; backend
        // verification still proves full protected membership and frozen states.
    }
    private struct Region {
        let nodes:[String:DataFlowNode]
        let edges:Set<DataFlowEdge>
    }
    private static func region(_ graph:DataFlowDocument,selected:String)throws->Region {
        let nodes=Dictionary(uniqueKeysWithValues:graph.nodes.map {($0.id,$0)})
        let incoming=Dictionary(grouping:graph.edges,by:{$0.to.node})
        var visited=Set<String>(),visiting=Set<String>()
        func visit(_ id:String)throws {
            if visited.contains(id) {return}
            guard !visiting.contains(id),nodes[id] != nil,visiting.count<128 else {
                throw failure("The selected test endpoint or a bounded acyclic ancestor is absent from archived preparation. Select an existing archived port; no raw-port substitution is performed.")
            }
            visiting.insert(id)
            for edge in incoming[id] ?? [] {try visit(edge.from.node)}
            visiting.remove(id);visited.insert(id)
        }
        try visit(selected)
        return .init(nodes:nodes.filter {visited.contains($0.key)},edges:Set(graph.edges.filter {visited.contains($0.to.node)}))
    }
    private static func failure(_ message:String)->DataFlowContractError {.init(message)}
}
