import Foundation

public enum CompositeExamples {
    /// The same two independent affine/nonlinear branches as the existing runnable example.
    public static var parallelClassifier:CompositeProjectDocument {
        let original=ExampleFlows.parallelClassifier
        let included:Set<String>=["branch-a","activation-a","branch-b","activation-b","merge"]
        let reference=CompositeDefinitionReference("kwip.parallel-features")
        let definition=CompositeDefinition(reference:reference,title:"Parallel features",body:CompositeGraph(nodes:original.nodes.filter {included.contains($0.id)},edges:original.edges.filter {included.contains($0.source) && included.contains($0.target)}),inputs:[CompositeInput("in",targets:[CompositePortReference("branch-a","in"),CompositePortReference("branch-b","in")])],outputs:[CompositeOutput("out",source:CompositePortReference("merge","out"))],parameters:[CompositeParameter(ComponentField(key:"branchWidth",title:"Branch width",defaultValue:8,min:1,max:512),targets:[CompositeParameterTarget("branch-a","out_features"),CompositeParameterTarget("branch-b","out_features")])])
        let root=CompositeGraph(nodes:original.nodes.filter {!included.contains($0.id)},instances:[CompositeInstance(id:"features",definition:reference,name:"Parallel features",x:300,y:205)],edges:[GraphEdge(id:"outer-in",source:"input",target:"features"),GraphEdge(id:"outer-out",source:"features",target:"classifier"),GraphEdge(id:"outer-result",source:"classifier",target:"output")])
        return CompositeProjectDocument(name:original.name,root:root,definitions:[definition],execution:original.execution)
    }
}
