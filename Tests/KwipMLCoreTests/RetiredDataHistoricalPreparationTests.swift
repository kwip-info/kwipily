import XCTest
@testable import KwipMLCore

final class RetiredDataHistoricalPreparationTests:XCTestCase {
    func testHistoricalPreparationAndCapturedCodeValidateOriginalRetiredGraphWithoutUpgrade()throws {
        let fixture=RetiredDataExecutionBoundaryTests(),(model,registry)=try fixture.registries()
        let source=WorkflowNodeAddress(domain:.data,localID:"source"),cast=WorkflowNodeAddress(domain:.data,localID:"cast")
        let config:[String:JSONValue]=["columns":.object(["x":.string("float64")]),"missingPolicy":.string("preserve")]
        let legacy=WorkflowDocument(projectID:UUID().uuidString,nodes:[
            .init(address:source,name:"Raw rows",payload:.dataOperator(operatorID:"kwip.data.sourceCSV",config:["separator":.string(",")])),
            .init(address:cast,name:"Retained cast",payload:.dataOperator(operatorID:"kwip.data.cast",config:config))],
            edges:[.init(from:.init(address:source,port:"table"),to:.init(address:cast,port:"table"),kind:.table)])
        let original=try WorkflowCodec.encode(legacy)
        let expected=DataFlowDocument(nodes:[.init(id:"source",operatorID:"kwip.data.sourceCSV",config:["separator":.string(",")]),.init(id:"cast",operatorID:"kwip.data.cast",config:config)],edges:[.init(from:.init(node:"source",port:"table"),to:.init(node:"cast",port:"table"))])
        XCTAssertEqual(try WorkflowDataPreparationPlan.compileHistoricalEvidence(legacy,registry:registry),expected)
        let captured=WorkflowCodePlan(schemaVersion:1,scope:.preparation,document:legacy,compilerContractVersion:12,modelRegistry:model,dataRegistry:registry,selectedTargets:[],preparation:.init(graph:expected,nodeLabels:["source":"Raw rows","cast":"Retained cast"],engine:"polars",sampling:nil,completeNodes:nil),models:[],trainings:[])
        let bytes=try captured.encoded(),decoded=try JSONDecoder().decode(WorkflowCodePlan.self,from:bytes)
        XCTAssertNoThrow(try decoded.validate(),"Captured Code reconstruction must use the historical preparation seam")
        XCTAssertEqual(try decoded.encoded(),bytes);XCTAssertEqual(try WorkflowCodec.encode(legacy),original)
        XCTAssertThrowsError(try WorkflowCodePlan.compile(legacy,scope:.preparation,modelRegistry:model,dataRegistry:registry))
    }
}
