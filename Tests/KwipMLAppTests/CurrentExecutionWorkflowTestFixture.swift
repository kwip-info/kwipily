import Foundation
import KwipMLCore

/// New execution fixtures use current authoring tools. This is an explicit test
/// scenario, not an inferred upgrade of historical two-way split membership.
enum CurrentExecutionWorkflowTestFixture {
    private static func address(_ domain:WorkflowDomain,_ id:String)->WorkflowNodeAddress {.init(domain:domain,localID:id)}
    private static func edge(_ a:WorkflowDomain,_ id:String,_ port:String,_ b:WorkflowDomain,_ target:String,_ input:String,_ kind:WorkflowEdgeKind)->WorkflowEdge {
        .init(id:"\(a)-\(id)-\(port)-\(b)-\(target)-\(input)",from:.init(address:address(a,id),port:port),to:.init(address:address(b,target),port:input),kind:kind)
    }
    static func columns(_ selections:[(String,String)])->[String:JSONValue] {
        ["rows":.array(selections.map {.object(["source":.string($0.0),"included":true,"outputName":.string($0.0),"targetType":.string($0.1)])}),"newColumns":.string("keep"),"castFailure":.string("error"),"trimText":false,"missingPolicy":.string("error")]
    }
    private static var scaling:[String:JSONValue] {
        PreprocessConfiguration(columns:["a","b"].map {name in var column=PreprocessConfiguration.Column(column:name);column.missing="error";column.scaling="standard";return column}).values
    }
    static func make(block:Bool=false)->WorkflowDocument {
        let base=ExampleFlows.parallelClassifier
        let source=CompositeExamples.parallelClassifier
        var nodes:[WorkflowNode]=[
            .init(address:address(.data,"input"),payload:.dataOperator(operatorID:"kwip.data.sourceCSV",config:["separator":.string(",")])),
            .init(address:address(.data,"cast"),payload:.dataOperator(operatorID:"kwip.data.columns",config:columns([("a","float64"),("b","float64"),("label","int64")]))),
            .init(address:address(.data,"split"),payload:.dataOperator(operatorID:"kwip.data.partition",config:["strategy":.string("random"),"seed":42,"validationPercent":20,"testPercent":20,"stratifyColumn":.string("label")])),
            .init(address:address(.data,"fit"),payload:.dataOperator(operatorID:"kwip.data.preprocess",config:scaling)),
            .init(address:address(.data,"training"),payload:.dataOperator(operatorID:"kwip.data.columns",config:columns([]))),
            .init(address:address(.data,"validation"),payload:.dataOperator(operatorID:"kwip.data.columns",config:columns([]))),
            .init(address:address(.control,"model"),payload:.modelDefinition),
            .init(address:address(.control,"mapping"),payload:.trainingData(config:["featureColumns":.array([.string("a"),.string("b")]),"targetColumn":.string("label")])),
            .init(address:address(.control,"setup"),payload:.runSetup(config:["device":.string("cpu"),"epochs":.integer(3),"batchSize":.integer(32),"seed":.integer(17),"learningRate":.number(0.02)])),
            .init(address:address(.control,"train"),payload:.train(config:[:]))]
        let primitives=block ? source.root.nodes : base.nodes
        nodes += primitives.map {.init(address:address(.model,$0.id),name:$0.name,x:$0.x,y:$0.y,payload:.modelPrimitive(type:$0.type,configurationVersion:block ? (source.root.configurationVersion ?? 1) : base.schemaVersion,config:$0.config))}
        if block {nodes += source.root.instances.map {.init(address:address(.model,$0.id),name:$0.name,x:$0.x,y:$0.y,payload:.modelBlock(reference:$0.definition,overrides:$0.overrides))}}
        var edges=[edge(.data,"input","table",.data,"cast","table",.table),edge(.data,"cast","table",.data,"split","table",.table),edge(.data,"split","training",.data,"fit","training",.table),edge(.data,"split","validation",.data,"fit","validation",.table),edge(.data,"split","test",.data,"fit","test",.table),edge(.data,"fit","training",.data,"training","table",.table),edge(.data,"fit","validation",.data,"validation","table",.table),edge(.data,"training","table",.control,"mapping","training",.table),edge(.data,"validation","table",.control,"mapping","validation",.table),edge(.model,"output","definition",.control,"model","source",.modelDefinition),edge(.control,"model","model",.control,"train","model",.modelDefinition),edge(.control,"mapping","data",.control,"train","data",.trainingData),edge(.control,"setup","configuration",.control,"train","configuration",.runConfiguration)]
        edges += (block ? source.root.edges : base.edges).map {.init(id:$0.id,from:.init(address:address(.model,$0.source),port:$0.sourcePort),to:.init(address:address(.model,$0.target),port:$0.targetPort),kind:.tensor)}
        return WorkflowDocument(projectID:UUID().uuidString,revision:7,name:"Measured workflow",nodes:nodes,edges:edges,definitions:block ? source.definitions : [])
    }
}
