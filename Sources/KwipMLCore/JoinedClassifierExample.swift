import Foundation

extension WorkflowExamples {
    public static let joinContextAddress=WorkflowNodeAddress(domain:.data,localID:"contextObservations")

    /// Independent observations joined one-to-one before any partition or learned transform.
    public static func preparedJoinedClassifier(projectID:String=UUID().uuidString,device:String="auto")throws->WorkflowDocument {
        var document=try preparedClassifier(projectID:projectID,device:device,model:nil)
        document.name="Join observations and train a classifier"
        let joined=WorkflowNodeAddress(domain:.data,localID:"join")
        let cast=WorkflowNodeAddress(domain:.data,localID:"cast")
        document.nodes.append(.init(address:joinContextAddress,name:"Context by observation",x:40,y:210,
            payload:.dataOperator(operatorID:"kwip.data.sourceCSV",config:["separator":.string(",")])))
        document.nodes.append(.init(address:joined,name:"Match observation identities",x:270,y:125,
            payload:.dataOperator(operatorID:"kwip.data.join",config:[
                "leftKeys":.array([.string("observation_id")]),"rightKeys":.array([.string("observation_id")]),
                "mode":.string("inner"),"nullPolicy":.string("error"),"cardinality":.string("oneToOne"),
                "leftPrefix":.string("signal_"),"rightPrefix":.string("context_"),"maxRows":.integer(1000)])))
        document.edges.removeAll {$0.from.address == sourceAddress && $0.to.address == cast}
        for (id,from,port,to,input) in [("join-left",sourceAddress,"table",joined,"left"),
                                      ("join-right",joinContextAddress,"table",joined,"right"),
                                      ("join-cast",joined,"table",cast,"table")] {
            document.edges.append(.init(id:id,from:.init(address:from,port:port),to:.init(address:to,port:input),kind:.table))
        }
        let data:[String:(Double,Double)]=["observations":(40,40),"contextObservations":(40,210),
            "join":(270,125),"cast":(500,125),"split":(730,125),"fit":(960,40),
            "training":(1190,40),"validation":(1190,210)]
        let model:[String:(Double,Double)]=["input":(40,420),"features":(270,420),"classifier":(500,420),"output":(730,420)]
        let control:[String:(Double,Double)]=["mapping":(1420,125),"model":(1190,420),"setup":(1420,420),"train":(1650,270)]
        for index in document.nodes.indices {
            let address=document.nodes[index].address
            let placements=address.domain == .data ? data : address.domain == .model ? model : control
            if let (x,y)=placements[address.localID] {document.nodes[index].x=x;document.nodes[index].y=y}
        }
        try WorkflowCodec.validate(document)
        return document
    }
}
