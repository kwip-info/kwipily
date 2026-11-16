import Foundation

public enum TypedDecisionPracticeExample {
 public static let exampleID="typed-decisions"
 public static let source=WorkflowNodeAddress(domain:.data,localID:"tickets")
 public static let train=WorkflowNodeAddress(domain:.control,localID:"train")
 public static let model=WorkflowNodeAddress(domain:.control,localID:"decision-model")
 public static let decision=WorkflowNodeAddress(domain:.data,localID:"decide")
 public static func document(projectID:String=UUID().uuidString)throws->WorkflowDocument {
  let partition=WorkflowNodeAddress(domain:.data,localID:"partition"),view=WorkflowNodeAddress(domain:.data,localID:"decisions"),scores=WorkflowNodeAddress(domain:.data,localID:"scores"),save=WorkflowNodeAddress(domain:.control,localID:"save-model")
  var load=LoadModelConfiguration();load.source = .huggingFace;load.repository="convaiinnovations/laya";load.revision="7b928d828b7b0e022f929d9bd2e44165aa270148";load.subfolder="typed-decisions"
  var decide=TypedDecisionConfiguration.defaults;decide["questionsFromModel"]=true;decide["identityColumn"] = .string("case_id")
  let optimizer=try OptimizerConfiguration(componentID:"optimizer_adamw",values:[:])
  let nodes:[WorkflowNode]=[
   .init(address:source,name:"Read support tickets",x:30,y:250,payload:.dataOperator(operatorID:"kwip.data.sourceCSV",config:CSVParsingConfiguration.newDefaults)),
   .init(address:partition,name:"Keep validation separate",x:350,y:250,payload:.dataOperator(operatorID:"kwip.data.partition",config:["strategy":.string("random"),"seed":23,"validationPercent":25,"testPercent":25])),
   .init(address:model,name:"Load typed-decision model",x:350,y:30,payload:.frozenModel(config:load.values)),
   .init(address:train,name:"Learn ticket decisions",x:680,y:100,payload:.train(config:["fineTuning":.object(TypedDecisionConfiguration.trainingDefaults),"optimizer":optimizer.recipeValue,"dataMapping":.object([:]),"runSetup":.object(["device":.string("cpu"),"epochs":1,"batchSize":1,"seed":23,"learningRate":0.0001])])),
   .init(address:decision,name:"Review held-out decisions",x:1010,y:250,payload:.dataOperator(operatorID:TypedDecisionConfiguration.operatorID,config:decide)),
   .init(address:view,name:"Inspect decisions",x:1340,y:150,payload:.dataOperator(operatorID:"kwip.data.view",config:[:])),
   .init(address:scores,name:"Inspect every option",x:1340,y:390,payload:.dataOperator(operatorID:"kwip.data.view",config:[:])),
   .init(address:save,name:"Save standalone decisions",x:1010,y:20,payload:.saveModel(config:WorkflowSaveModelConfiguration(format:.huggingFace).values))]
  func table(_ a:WorkflowNodeAddress,_ port:String,_ b:WorkflowNodeAddress,_ input:String="table")->WorkflowEdge {.init(from:.init(address:a,port:port),to:.init(address:b,port:input),kind:.table)}
  func resource(_ a:WorkflowNodeAddress,_ b:WorkflowNodeAddress)->WorkflowEdge {.init(from:.init(address:a,port:"model"),to:.init(address:b,port:"model"),kind:.frozenModel)}
  var doc=WorkflowDocument(projectID:projectID,name:"Train typed decisions for support tickets",nodes:nodes,edges:[table(source,"table",partition),table(partition,"training",train,"training"),table(partition,"validation",train,"validation"),table(partition,"validation",decision),resource(model,train),resource(train,decision),resource(train,save),table(decision,"decisions",view),table(decision,"scores",scores)])
  doc.schemaVersion=21;try WorkflowCodec.validate(doc);return doc
 }
}
