import KwipMLCore

enum FormulaSampleScope {
    static func permits(consumer:WorkflowNodeAddress,input:WorkflowEndpoint,document:WorkflowDocument,selection:WorkflowSelection)->Bool {
        guard selection.nodeAddresses == [consumer],selection.edgeIDs.isEmpty,
              let node=document.nodes.first(where:{$0.address == consumer}),
              case .dataOperator(let id,_)=node.payload,id == "kwip.data.formula" else {return false}
        let edges=document.edges.filter{$0.to.address == consumer && $0.to.port == "table"}
        return edges.count == 1 && edges[0].from == input && [.table,.dataset].contains(edges[0].kind)
    }
}
