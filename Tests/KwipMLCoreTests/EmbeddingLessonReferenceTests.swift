import XCTest
@testable import KwipMLCore

final class EmbeddingLessonReferenceTests:XCTestCase {
    private func config(_ changes:[String:JSONValue]=[:])throws->EmbeddingConfiguration {
        var values=Dictionary(uniqueKeysWithValues:EmbeddingComponent.definition.fields.map {($0.key,$0.defaultValue)})
        values["num_embeddings"]=3;values["embedding_dim"]=2
        values.merge(changes){_,new in new};return try EmbeddingConfiguration(values:values)
    }
    func testGatherAndRepeatedIndexScatterWithPaddingAndFrequency()throws {
        let table=LearningTensor(shape:[3,2],values:[9,8,1,2,3,4])
        for frequency in [false,true] {
            let reference=try EmbeddingLessonReference(indices:[0,2,2,1],shape:[2,2],table:table,configuration:config(["padding_idx":0,"scale_grad_by_freq":.bool(frequency)]))
            XCTAssertEqual(reference.output.shape,[2,2,2])
            XCTAssertEqual(reference.output.values,[9,8,3,4,3,4,1,2])
            let gradient=try XCTUnwrap(reference.tableGradient(for:.init(shape:[2,2,2],values:[1,2,3,4,5,6,7,8])))
            XCTAssertEqual(gradient.values,frequency ? [0,0,7,8,4,5] : [0,0,7,8,8,10])
            XCTAssertEqual(reference.tableAfterForward,table)
        }
    }
    func testFrozenLookupStillRenormalizesSelectedRowsAndOwnsNoGradient()throws {
        let table=LearningTensor(shape:[3,2],values:[3,4,6,8,5,12])
        let reference=try EmbeddingLessonReference(indices:[1,1],shape:[1,2],table:table,configuration:config(["freeze":true,"max_norm":1]))
        XCTAssertEqual(reference.tableAfterForward.values[0...1],[3,4])
        XCTAssertEqual(reference.tableAfterForward.values[4...5],[5,12])
        XCTAssertEqual(reference.output.values[0],6/(10+1e-7),accuracy:1e-12)
        XCTAssertEqual(reference.output.values[2],reference.output.values[0])
        XCTAssertNil(try reference.tableGradient(for:.init(shape:[1,2,2],values:[1,1,1,1])))
    }
    func testOrdinaryGatherGradientMatchesFiniteDifferencesForEveryTableCoordinate()throws {
        let table=LearningTensor(shape:[3,2],values:[0.5,-1,2,3,-0.25,4])
        let upstream=LearningTensor(shape:[1,3,2],values:[1,-2,3,4,-5,6])
        let configuration=try config()
        let reference=try EmbeddingLessonReference(indices:[2,0,2],shape:[1,3],table:table,configuration:configuration)
        let gradient=try XCTUnwrap(reference.tableGradient(for:upstream))
        func loss(_ table:LearningTensor)throws->Double {
            let output=try EmbeddingLessonReference(indices:[2,0,2],shape:[1,3],table:table,configuration:configuration).output
            return zip(output.values,upstream.values).reduce(0) {$0+$1.0*$1.1}
        }
        let step=1e-6
        for index in table.values.indices {
            var plus=table,minus=table
            plus.values[index]+=step;minus.values[index]-=step
            XCTAssertEqual(gradient.values[index],try (loss(plus)-loss(minus))/(2*step),accuracy:1e-8)
        }
    }
    func testRenormalizationUsesConfiguredNormAndIncludesSelectedPaddingRow()throws {
        let table=LearningTensor(shape:[3,2],values:[3,4,1,2,5,12])
        for (order,norm) in [(1.0,7.0),(0.5,pow(sqrt(3.0)+2,2))] {
            let reference=try EmbeddingLessonReference(indices:[0,1],shape:[1,2],table:table,configuration:config(["padding_idx":0,"max_norm":2,"norm_type":.number(order)]))
            XCTAssertEqual(reference.output.values[0],3*2/(norm+1e-7),accuracy:1e-12)
            XCTAssertEqual(reference.output.values[1],4*2/(norm+1e-7),accuracy:1e-12)
            XCTAssertEqual(Array(reference.tableAfterForward.values.suffix(2)),[5,12])
            let gradient=try XCTUnwrap(reference.tableGradient(for:.init(shape:[1,2,2],values:[9,9,2,3])))
            XCTAssertEqual(gradient.values,[0,0,2,3,0,0])
        }
    }
    func testSparseStorageDoesNotChangeGatherOrMathematicalScatter()throws {
        let table=LearningTensor(shape:[3,2],values:[0,1,2,3,4,5])
        let reference=try EmbeddingLessonReference(indices:[2,2],shape:[1,2],table:table,configuration:config(["sparse":true]))
        XCTAssertEqual(reference.output.values,[4,5,4,5])
        XCTAssertEqual(try reference.tableGradient(for:.init(shape:[1,2,2],values:[1,2,3,4]))?.values,[0,0,0,0,4,6])
        XCTAssertThrowsError(try EmbeddingLessonReference(indices:[2],shape:[1,1],table:table,configuration:config(["sparse":true,"scale_grad_by_freq":true])))
    }
    func testInvalidIndexShapeAndNonfiniteGradientReject()throws {
        let table=LearningTensor(shape:[3,2],values:[0,1,2,3,4,5])
        for indices:[Int64] in [[-1],[3],[Int64.max]] {
            XCTAssertThrowsError(try EmbeddingLessonReference(indices:indices,shape:[1,1],table:table,configuration:config()))
        }
        XCTAssertThrowsError(try EmbeddingLessonReference(indices:[1],shape:[Int.max,2],table:table,configuration:config()))
        let reference=try EmbeddingLessonReference(indices:[1],shape:[1,1],table:table,configuration:config())
        XCTAssertThrowsError(try reference.tableGradient(for:.init(shape:[1,1,2],values:[.infinity,0])))
        XCTAssertThrowsError(try reference.tableGradient(for:.init(shape:[2],values:[1,2])))
    }
}
