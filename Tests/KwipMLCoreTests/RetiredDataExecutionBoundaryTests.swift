import XCTest
@testable import KwipMLCore

final class RetiredDataExecutionBoundaryTests:XCTestCase {
    func registries()throws->(ComponentRegistry,DataFlowOperatorRegistry) {
        let root=URL(fileURLWithPath:#filePath).deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
        var captured=try DataFlowOperatorRegistry.load(from:root.appendingPathComponent("runner/dataflow-operators.json"))
        let retired=try DataFlowOperatorRegistry.load(from:root.appendingPathComponent("runner/retired-data-operators-v1.json"))
        XCTAssertTrue(Set(captured.operators.map(\.id)).isDisjoint(with:retired.operators.map(\.id)))
        captured.operators += retired.operators
        return(try .builtIn(),captured)
    }
    func legacyTrain()throws->WorkflowDocument {
        var document=try WorkflowExamples.preparedParallelClassifier(device:"cpu")
        let index=try XCTUnwrap(document.nodes.firstIndex {$0.address == .init(domain:.data,localID:"cast")})
        document.nodes[index].payload = .dataOperator(operatorID:"kwip.data.cast",config:["columns":.object(["signal":.string("float64"),"context":.string("float64"),"label":.string("int64")]),"missingPolicy":.string("error")])
        return document
    }
    func assertRetired(_ body:()throws->Void,file:StaticString=#filePath,line:UInt=#line) {
        XCTAssertThrowsError(try body(),file:file,line:line) {error in
            XCTAssertTrue(error.localizedDescription.lowercased().contains("retired data"),error.localizedDescription,file:file,line:line)
        }
    }
    func testFreshTrainingRefusesRetiredDataEvenWithMatchingCapturedRegistry()throws {
        let(model,registry)=try registries(),source=try legacyTrain()
        assertRetired {_ = try WorkflowCompiler.compile(source,train:WorkflowExamples.trainAddress,modelRegistry:model,dataRegistry:registry,contractVersion:12)}
    }
    func testFreshPreparationRefusesRetiredDataEvenWithMatchingCapturedRegistry()throws {
        let(_,registry)=try registries(),source=try legacyTrain()
        assertRetired {_ = try WorkflowDataPreparationPlan.compile(source,registry:registry,allowTrainingTerminals:true)}
    }
    func testFreshCodeRefusesRetiredDataEvenWithMatchingCapturedRegistry()throws {
        let(model,registry)=try registries(),source=try legacyTrain()
        for scope in [WorkflowCodePlan.Scope.workflow,.preparation] {
            assertRetired {_ = try WorkflowCodePlan.compile(source,scope:scope,modelRegistry:model,dataRegistry:registry)}
        }
    }
    func testCurrentEquivalentStillCompilesAndHistoricalTrainingPreservesOriginalBytes()throws {
        let(model,registry)=try registries(),legacy=try legacyTrain(),original=try WorkflowCodec.encode(legacy)
        let historical=try WorkflowCompiler.compileHistoricalEvidence(legacy,train:WorkflowExamples.trainAddress,modelRegistry:model,dataRegistry:registry,contractVersion:12)
        XCTAssertEqual(historical.dataflow.nodes.first {$0.id=="cast"}?.operatorID,"kwip.data.cast")
        XCTAssertEqual(try WorkflowCodec.encode(legacy),original)
        let current=try WorkflowExamples.preparedParallelClassifier(device:"cpu")
        let plan=try WorkflowCompiler.compile(current,train:WorkflowExamples.trainAddress,modelRegistry:model,dataRegistry:registry,contractVersion:12)
        XCTAssertEqual(plan.model,historical.model)
        XCTAssertNoThrow(try WorkflowDataPreparationPlan.compile(current,registry:registry,allowTrainingTerminals:true))
        XCTAssertNoThrow(try WorkflowCodePlan.compile(current,modelRegistry:model,dataRegistry:registry))
    }
}
