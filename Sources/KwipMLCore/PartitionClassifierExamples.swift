import Foundation

extension WorkflowExamples {
    public static func preparedGroupSplitClassifier(projectID:String=UUID().uuidString,device:String="auto")throws->WorkflowDocument {
        try partitionClassifier(projectID:projectID,device:device,temporal:false)
    }
    public static func preparedTemporalSplitClassifier(projectID:String=UUID().uuidString,device:String="auto")throws->WorkflowDocument {
        try partitionClassifier(projectID:projectID,device:device,temporal:true)
    }
    private static func partitionClassifier(projectID:String,device:String,temporal:Bool)throws->WorkflowDocument {
        var document=try preparedClassifier(projectID:projectID,device:device,model:nil)
        document.name=temporal ? "Chronological holdouts and classifier" : "Keep subjects together and train"
        guard let index=document.nodes.firstIndex(where:{$0.address.localID == "split" && $0.address.domain == .data}) else {throw DataFlowContractError("Example split is missing.")}
        let config:[String:JSONValue]
        if temporal {
            config=["strategy":.string("time"),"timeColumn":.string("observed_at"),"timeEncoding":.string("iso8601Offset"),
                "trainingEnd":.string("2026-01-02T23:00:00Z"),"validationEnd":.string("2026-01-03T23:00:00Z"),
                "embargoMilliseconds":.integer(0),"entityColumns":.array([.string("subject_id")]),"entityPolicy":.string("requireDisjoint")]
        } else {
            config=["strategy":.string("entity"),"groupBy":.array([.string("subject_id")]),"seed":.integer(42),"validationPercent":.integer(25),"testPercent":.integer(25)]
        }
        document.nodes[index].name=temporal ? "Separate past, validation and future" : "Reserve independent subjects"
        document.nodes[index].payload = .dataOperator(operatorID:"kwip.data.partition",config:config)
        if let source=document.nodes.firstIndex(where:{$0.address == sourceAddress}) {
            document.nodes[source].name=temporal ? "Dated independent observations" : "Repeated subject measurements"
        }
        // The protected branch is an explicit, separately launched evaluation.
        // Checkpoint selection stays private and unbound until after training.
        document.schemaVersion=max(document.schemaVersion,5)
        let frozen=WorkflowNodeAddress(domain:.control,localID:"heldout-frozen-model")
        let evaluate=WorkflowNodeAddress(domain:.control,localID:"heldout-evaluation")
        document.nodes += [
            .init(address:frozen,name:"Choose a completed checkpoint",x:860,y:1010,payload:.frozenModel(config:[:])),
            .init(address:evaluate,name:"Evaluate protected observations",x:1120,y:1010,
                  payload:.evaluateTest(config:["device":.string(device),"batchSize":.integer(32),"previewLimit":.integer(8)]))
        ]
        document.edges += [
            .init(from:.init(address:frozen,port:"model"),to:.init(address:evaluate,port:"model"),kind:.frozenModel),
            .init(from:.init(address:document.nodes[index].address,port:"test"),to:.init(address:evaluate,port:"test"),kind:.table)
        ]
        try WorkflowCodec.validate(document)
        return document
    }
}
