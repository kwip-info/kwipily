import Foundation

extension WorkflowExamples {
    public static func preparedEmbeddingBagClassifier(projectID:String = UUID().uuidString,device:String = "auto") throws -> WorkflowDocument {
        var source=BagSourceConfiguration.defaults
        source["maximum_items_per_bag"]=3
        var table=EmbeddingBagConfiguration.defaults
        table["norm_type"]=2 // Canonical JSON integer token preserves captured document equality.
        let graph=GraphDocument(schemaVersion:2,name:"Grouped category classifier",nodes:[
            .init(id:"input",type:"bag_input",name:"Three categories per observation",config:source),
            .init(id:"lookup",type:"embedding_bag",name:"Combine category vectors",config:table),
            .init(id:"head",type:"linear",name:"Compare two classes",config:["in_features":4,"out_features":2,"bias":true]),
            .init(id:"output",type:"output",name:"Class scores",config:[:])
        ],edges:[
            .init(id:"categories",source:"input",sourcePort:"indices",target:"lookup",targetPort:"indices"),
            .init(id:"combined",source:"lookup",target:"head"),
            .init(id:"scores",source:"head",target:"output")
        ])
        var document=try preparedClassifier(projectID:projectID,device:device,model:graph,scaleFeatures:false)
        document.schemaVersion=12
        document.name="Prepare and train an embedding bag classifier"
        let mapping:JSONValue = .object(["schemaVersion":1,"indicesColumns":.array(["id_a","id_b","id_c"].map(JSONValue.string))])
        for index in document.nodes.indices {
            let address=document.nodes[index].address
            if address == sourceAddress {document.nodes[index].name="Grouped category observations"}
            if address.domain == .data,address.localID == "cast" {
                document.nodes[index].name="Keep category IDs as whole numbers"
                document.nodes[index].payload = .dataOperator(operatorID:"kwip.data.columns",config:CurrentPracticeConfiguration.columns([("id_a","int64"),("id_b","int64"),("id_c","int64"),("label","int64")]))
            }
            if address.domain == .control,address.localID == "mapping" {
                document.nodes[index].name="Whole bags and class target"
                document.nodes[index].payload = .trainingData(config:["bagInput":mapping,"targetColumn":.string("label")])
            }
        }
        let lanes:[(WorkflowDomain,[String],Double)]=[
            (.data,["observations","cast","split"],40),
            (.model,["input","lookup","head","output"],230),
            (.control,["mapping","model","setup","train"],420)]
        for (domain,ids,y) in lanes {
            for (column,id) in ids.enumerated() {
                if let index=document.nodes.firstIndex(where:{$0.address == .init(domain:domain,localID:id)}) {
                    document.nodes[index].x=40+Double(column)*240;document.nodes[index].y=y
                }
            }
        }
        try WorkflowCodec.validate(document)
        return document
    }
}
