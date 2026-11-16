import XCTest
@testable import KwipMLCore

final class DatasetLoaderWorkflowCodecTests:XCTestCase {
    private func bytes(_ id:String,version:Int=13,extra:Bool=false)throws->Data {
        let dataset:[String:Any] = ["kind":"indexed","indices":[],"max_rows":100000]
        let loader:[String:Any] = ["batch_size":32,"shuffle":false,"drop_last":false,"seed":42,"num_workers":0,"prefetch_factor":NSNull(),"timeout":0.0,"collation":"auto"]
        var config = id == "dataset_source" ? dataset:loader
        if extra {config["foreign"] = true}
        let payload:[String:Any] = ["kind":id == "dataset_source" ? "datasetRecipe":"batchRecipe","type":id,"configurationVersion":1,"config":config]
        let root:[String:Any] = ["schemaVersion":version,"kind":"workflowProject","projectID":UUID().uuidString,"revision":0,"name":"Choose observations and groups","nodes":[["address":["domain":"control","localID":"reader"],"name":"My reader","x":0,"y":0,"payload":payload]],"edges":[],"definitions":[],"drafts":[]]
        return try JSONSerialization.data(withJSONObject:root)
    }
    func testBothRecipeKindsRoundTripWithQualifiedSource()throws {
        for id in ["dataset_source","data_loader"] {
            let doc=try WorkflowCodec.decode(bytes(id))
            XCTAssertEqual(doc.schemaVersion,13)
            XCTAssertEqual(doc.nodes.first?.address,.init(domain:.control,localID:"reader"))
            XCTAssertEqual(try WorkflowCodec.decode(WorkflowCodec.encode(doc)),doc)
        }
    }
    func testOlderSchemaAndUnknownConfigurationRefuse()throws {
        for id in ["dataset_source","data_loader"] {
            for version in 1...12 {XCTAssertThrowsError(try WorkflowCodec.decode(bytes(id,version:version)))}
            XCTAssertThrowsError(try WorkflowCodec.decode(bytes(id,extra:true)))
        }
        let old=WorkflowDocument(projectID:UUID().uuidString)
        XCTAssertEqual(try WorkflowCodec.decode(WorkflowCodec.encode(old)),old)
    }
}
