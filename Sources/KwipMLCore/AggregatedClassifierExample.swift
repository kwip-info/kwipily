import Foundation

extension WorkflowExamples {
    /// One output per independent subject before partitioning or fitting statistics.
    /// Labels in this bundled fixture are consistent; min/max expose, not repair, conflicts.
    public static func preparedAggregatedClassifier(projectID:String=UUID().uuidString,device:String="auto")throws->WorkflowDocument {
        var document=try preparedClassifier(projectID:projectID,device:device,model:nil)
        document.name="Aggregate repeated measurements and train"
        let aggregate=WorkflowNodeAddress(domain:.data,localID:"aggregate")
        let cast=WorkflowNodeAddress(domain:.data,localID:"cast")
        let split=WorkflowNodeAddress(domain:.data,localID:"split")
        func reduction(_ op:String,_ column:String,_ output:String)->JSONValue {
            .object(["op":.string(op),"column":.string(column),"output":.string(output),"nullPolicy":.string("error")])
        }
        document.nodes.append(.init(address:aggregate,name:"One observation per subject",x:500,y:40,
            payload:.dataOperator(operatorID:"kwip.data.aggregate",config:[
                "groupBy":.array([.string("subject_id")]),"nullKeys":.string("error"),"maxRows":.integer(1000),
                "aggregations":.array([reduction("mean","signal","signal"),reduction("mean","context","context"),
                    reduction("min","label","label"),reduction("max","label","label_check"),
                    .object(["op":.string("countRows"),"output":.string("measurement_count")])])])))
        document.edges.removeAll {$0.from.address == cast && $0.to.address == split}
        for (id,from,to) in [("aggregate-cast",cast,aggregate),("aggregate-split",aggregate,split)] {
            document.edges.append(.init(id:id,from:.init(address:from,port:"table"),to:.init(address:to,port:"table"),kind:.table))
        }
        let data:[String:(Double,Double)]=["observations":(40,40),"cast":(270,40),"aggregate":(500,40),
            "split":(730,40),"fit":(960,40),"training":(1190,40),"validation":(1190,210)]
        let model:[String:(Double,Double)]=["input":(40,420),"features":(270,420),"classifier":(500,420),"output":(730,420)]
        let control:[String:(Double,Double)]=["mapping":(1420,125),"model":(1190,420),"setup":(1420,420),"train":(1650,270)]
        for index in document.nodes.indices {
            let address=document.nodes[index].address
            if address == sourceAddress {document.nodes[index].name="Three measurements per subject"}
            let placements=address.domain == .data ? data : address.domain == .model ? model : control
            if let (x,y)=placements[address.localID] {document.nodes[index].x=x;document.nodes[index].y=y}
        }
        try WorkflowCodec.validate(document)
        return document
    }
}
