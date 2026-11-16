import XCTest
@testable import KwipMLCore

final class SequenceTrainingTests:XCTestCase {
    private func compiled()throws->(WorkflowDocument,WorkflowExecutionPlan) {
        let root=URL(fileURLWithPath:#filePath).deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
        let data=try DataFlowOperatorRegistry.load(from:root.appendingPathComponent("runner/dataflow-operators.json"))
        let d=try SequenceTrainingPracticeExample.document()
        return (d,try WorkflowCompiler.compile(d,train:WorkflowExamples.trainAddress,modelRegistry:ComponentRegistry.builtIn(),dataRegistry:data,contractVersion:15))
    }
    func testSequencePracticeCompilesPersistsAndRetainsOriginalTokenEvidence()throws {
        let (d,p)=try compiled()
        XCTAssertEqual(try WorkflowCodec.decode(WorkflowCodec.encode(d)),d)
        let named=try XCTUnwrap(p.namedInputPlan)
        XCTAssertEqual(try NamedInputPlan(values:named.values),named)
        XCTAssertEqual(named.schemaVersion,2);XCTAssertEqual(named.sequenceObjective?.maskColumn,"loss_mask")
        XCTAssertEqual(p.requiredPreparationNodes,["text","windows"])
        XCTAssertEqual(p.lossRecipe?.targetMapping.schemaVersion,2)
        XCTAssertEqual(try SequenceTrainingContract.predictionRowLimit(plan:named,outputShape:[3,4],count:16),8)
        var bad=named;bad.roles[0].lengthPolicy="pad";XCTAssertThrowsError(try bad.validate())
        bad=named;bad.sequenceObjective?.lengthColumn="input_ids";XCTAssertThrowsError(try bad.validate())
        var raw=named.values;raw["surprise"]=true;XCTAssertThrowsError(try NamedInputPlan(values:raw))
        XCTAssertThrowsError(try SequenceTrainingContract.predictionRowLimit(plan:named,outputShape:[4096,512],count:10))
    }
    func testSequenceObjectiveRequiresCanonicalValidTokenMean()throws {
        let (_,p)=try compiled(),recipe=try XCTUnwrap(p.lossRecipe)
        XCTAssertNoThrow(try recipe.configuration.validate(targetMapping:recipe.targetMapping))
        for values:[String:JSONValue] in [["label_smoothing":0.1],["reduction":.string("sum")],["ignore_index":-1]] {
            let changed=try LossConfiguration(componentID:"loss_cross_entropy",values:values)
            XCTAssertThrowsError(try changed.validate(targetMapping:recipe.targetMapping))
        }
    }
}
