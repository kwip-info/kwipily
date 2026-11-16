import Foundation

/// Synthetic local documents only. Resource/source grants remain private project bindings.
public enum DocumentClassificationPracticeExample {
    public static let exampleID="document-classification"
    public static let source=WorkflowNodeAddress(domain:.data,localID:"documents")
    public static let labels=WorkflowNodeAddress(domain:.data,localID:"labels")
    public static let features=WorkflowNodeAddress(domain:.data,localID:"features")
    public static let estimator=WorkflowNodeAddress(domain:.control,localID:"classifier")
    public static let predict=WorkflowNodeAddress(domain:.control,localID:"predict")
    public static let unseen=WorkflowNodeAddress(domain:.data,localID:"unseen")
    public static var documents:[(String,Data)] {
        let categories=[("invoice","Invoice payment total balance due. Please pay the attached invoice."),("contract","Contract agreement terms signed legal parties. This agreement governs the contract."),("support","Support help login account problem. Please help reset access to the account.")]
        return categories.flatMap {kind,text in (1...12).map {(String(format:"%@-%02d.txt",kind,$0),Data((text+" Reference document "+String($0)+".").utf8))}}
    }
    public static func document(projectID:String=UUID().uuidString)throws->WorkflowDocument {
        var files=FileInputConfiguration.newDefaults;files["sourceMode"] = .string("directory");files["extensions"] = .array([.string("txt")]);files["readMode"] = .string("content")
        var text:[String:JSONValue]=["textColumn":.string("text_content"),"tokenizer":.string("character"),"unicodeNormalization":.string("none"),"letterCase":.string("preserve"),"whitespace":.string("preserve"),"missingText":.string("error"),"outputColumn":.string("features"),"keepSourceText":true,"vocabularySource":.string("training"),"vocabulary":.array([]),"minimumFrequency":1,"vocabularyLimit":256,"idOrder":.string("frequency"),"unknownPolicy":.string("unk"),"pad":true,"bos":false,"eos":false]
        text["mode"] = .string("documentFeatures");text["featureSettings"] = .object(DocumentClassificationConfiguration.featureDefaults)
        var fit=DocumentClassificationConfiguration.defaults;fit["featureColumns"] = .array([.string("text_content")])
        var prediction=fit;prediction["operation"] = .string("predict")
        let joined=WorkflowNodeAddress(domain:.data,localID:"join"),partition=WorkflowNodeAddress(domain:.data,localID:"partition"),save=WorkflowNodeAddress(domain:.control,localID:"save"),load=WorkflowNodeAddress(domain:.control,localID:"load"),view=WorkflowNodeAddress(domain:.data,localID:"predictions")
        let nodes:[WorkflowNode]=[
            .init(address:source,name:"Read private documents",x:40,y:70,payload:.dataOperator(operatorID:DataFlowSourceOperators.fileInput,config:files)),
            .init(address:labels,name:"Labels by document",x:40,y:290,payload:.dataOperator(operatorID:DataFlowSourceOperators.tableInput,config:CSVParsingConfiguration.newDefaults)),
            .init(address:joined,name:"Match each document to its label",x:350,y:150,payload:.dataOperator(operatorID:"kwip.data.join",config:["leftKeys":.array([.string("relative_path")]),"rightKeys":.array([.string("relative_path")]),"mode":.string("left"),"nullPolicy":.string("error"),"cardinality":.string("oneToOne"),"leftPrefix":.string("document_"),"rightPrefix":.string("label_"),"maxRows":1000])),
            .init(address:partition,name:"Hold out documents",x:660,y:150,payload:.dataOperator(operatorID:"kwip.data.partition",config:["strategy":.string("random"),"seed":42,"validationPercent":20,"testPercent":20,"stratifyColumn":.string("category")])),
            .init(address:features,name:"Learn Training-only text features",x:970,y:150,payload:.dataOperator(operatorID:TextEncodingConfiguration.operatorID,config:text)),
            .init(address:estimator,name:"Fit document classifier",x:1280,y:150,payload:.estimator(config:fit)),
            .init(address:save,name:"Save classifier and text preparation",x:1590,y:150,payload:.saveModel(config:[:])),
            .init(address:load,name:"Reload classifier package",x:40,y:660,payload:.frozenModel(config:LoadModelConfiguration(source:.documentPackage).values)),
            .init(address:unseen,name:"Three unseen documents",x:40,y:460,payload:.dataOperator(operatorID:DataFlowSourceOperators.tableInput,config:CSVParsingConfiguration.newDefaults)),
            .init(address:predict,name:"Classify unseen documents",x:660,y:560,payload:.estimator(config:prediction)),
            .init(address:view,name:"Inspect document predictions",x:970,y:560,payload:.dataOperator(operatorID:"kwip.data.view",config:[:]))]
        func table(_ id:String,_ a:WorkflowNodeAddress,_ ap:String,_ b:WorkflowNodeAddress,_ bp:String)->WorkflowEdge {.init(id:id,from:.init(address:a,port:ap),to:.init(address:b,port:bp),kind:.table)}
        var edges=[table("documents-join",source,"table",joined,"left"),table("labels-join",labels,"table",joined,"right"),table("join-partition",joined,"table",partition,"table")]
        for role in ["training","validation","test"] {edges.append(table("features-"+role,partition,role,features,role))}
        for role in ["training","validation"] {edges.append(table("classifier-"+role,features,role,estimator,role))}
        edges += [table("unseen-predict",unseen,"table",predict,"table"),table("predict-view",predict,"predictions",view,"table"),.init(id:"classifier-save",from:.init(address:estimator,port:"model"),to:.init(address:save,port:"model"),kind:.frozenModel),.init(id:"loaded-predict",from:.init(address:load,port:"model"),to:.init(address:predict,port:"model"),kind:.frozenModel)]
        var document=WorkflowDocument(projectID:projectID,name:"Classify private documents",nodes:nodes,edges:edges);document.schemaVersion=21
        try WorkflowCodec.validate(document);return document
    }
}
