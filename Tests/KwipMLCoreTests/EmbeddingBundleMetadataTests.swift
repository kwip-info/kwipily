import XCTest
@testable import KwipMLCore

final class EmbeddingBundleMetadataTests:XCTestCase {
    func testLookupAndIntegerBoundaryRetainExactAuthoredContractsAndActualLessons()throws {
        let metadata=try XCTUnwrap(EmbeddingComponent.metadata)
        let input=try XCTUnwrap(EmbeddingComponent.indexInputMetadata)
        XCTAssertEqual(metadata.definition,EmbeddingComponent.definition)
        XCTAssertEqual(input.definition,EmbeddingComponent.indexInputDefinition)
        let additiveIDs:Set<String>=["dataset_source","data_loader"]
        XCTAssertEqual(fixtureModelComponentsBeforeTextExtensions(try .builtIn()).filter { $0.id != "initialization" && $0.id != "gradient_policy" && $0.id != "trainability" && !additiveIDs.contains($0.id) }.count,85)
        let currentComponents=fixtureModelComponentsBeforeTextExtensions(try .builtIn())
        XCTAssertEqual(Set(currentComponents.filter {additiveIDs.contains($0.id)}.map(\.id)),additiveIDs)
        XCTAssertEqual(currentComponents.filter { !["initialization","gradient_policy","trainability"].contains($0.id) }.count,87)
        XCTAssertEqual(metadata.presentation.icon,"embedding-v2")
        XCTAssertEqual(input.presentation.icon,"index_input-v2")
        XCTAssertEqual(metadata.definition.fields.map(\.key),["num_embeddings","embedding_dim","padding_idx","max_norm","norm_type","scale_grad_by_freq","sparse","freeze"])
        XCTAssertEqual(metadata.learning?.exampleID,"prepared-embedding-classifier")
        let descriptor=try BuiltInCapabilityBridge.embeddingDeclaration(metadata.definition)
        XCTAssertEqual(descriptor.learning.lessonIDs,["assessment.embedding-gather","assessment.embedding-padding-frequency","assessment.embedding-renormalize"])
        XCTAssertEqual(descriptor.adapterID,"kwip.runner.graph-v2.embedding")
        XCTAssertEqual(descriptor.lifecycle.state.count,1)
        XCTAssertTrue(try BuiltInCapabilityBridge.embeddingDeclaration(input.definition).lifecycle.state.isEmpty)
        for original in [metadata.definition,input.definition] {
            for key in ["id","title","category"] {
                var wrong=original
                if key == "id" {wrong.id="wrong"}
                if key == "title" {wrong.title="wrong"}
                if key == "category" {wrong.category="wrong"}
                XCTAssertThrowsError(try BuiltInCapabilityBridge.embeddingDeclaration(wrong),key)
            }
        }
    }
}
