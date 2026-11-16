import Foundation

public enum OCRPracticeExample {
    public static let exampleID="extract-text"
    public static let source=WorkflowNodeAddress(domain:.data,localID:"scans")
    public static let extract=WorkflowNodeAddress(domain:.data,localID:"ocr")
    public static let resourceNames=["ocr-invoice.png","ocr-contract.png","ocr-support.png"]
    public static func document(projectID:String=UUID().uuidString)throws->WorkflowDocument {
        var files=FileInputConfiguration.newDefaults;files["sourceMode"] = .string("directory");files["extensions"] = .array([.string("png")])
        var prepare=try PrepareMediaConfiguration.defaults();prepare["outputMode"] = .string("image");prepare["outputColumn"] = .string("ocr_image");prepare["color"] = .string("RGB");prepare["alpha"] = .string("composite");prepare["matte"] = .array([1,1,1])
        var ocr=try OCRConfiguration.defaults();ocr["inputColumn"] = .string("ocr_image")
        let pixels=WorkflowNodeAddress(domain:.data,localID:"pixels"),documents=WorkflowNodeAddress(domain:.data,localID:"documents"),regions=WorkflowNodeAddress(domain:.data,localID:"regions")
        let nodes:[WorkflowNode]=[
            .init(address:source,name:"Read synthetic scans",x:40,y:100,payload:.dataOperator(operatorID:DataFlowSourceOperators.fileInput,config:files)),
            .init(address:pixels,name:"Prepare OCR images",x:350,y:100,payload:.dataOperator(operatorID:PrepareMediaConfiguration.operatorID,config:prepare)),
            .init(address:extract,name:"Read printed text",x:660,y:100,payload:.dataOperator(operatorID:OCRConfiguration.operatorID,config:ocr)),
            .init(address:documents,name:"Inspect document text",x:990,y:40,payload:.dataOperator(operatorID:"kwip.data.view",config:[:])),
            .init(address:regions,name:"Inspect text regions",x:990,y:310,payload:.dataOperator(operatorID:"kwip.data.view",config:[:]))]
        func edge(_ id:String,_ a:WorkflowNodeAddress,_ port:String,_ b:WorkflowNodeAddress)->WorkflowEdge {.init(id:id,from:.init(address:a,port:port),to:.init(address:b,port:"table"),kind:.table)}
        let document=WorkflowDocument(projectID:projectID,name:"Read text from scanned images",nodes:nodes,edges:[edge("scan-pixels",source,"table",pixels),edge("pixels-ocr",pixels,"table",extract),edge("ocr-documents",extract,"documents",documents),edge("ocr-regions",extract,"regions",regions)])
        try WorkflowCodec.validate(document);return document
    }
}
