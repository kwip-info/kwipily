import XCTest
import KwipMLCore
@testable import KwipMLApp

@MainActor final class LossArtworkTests:XCTestCase {
    private let ids:Set<String>=["loss_cross_entropy","loss_bce_with_logits","loss_mse","loss_l1","loss_nll","loss_kl_div","loss_ctc","loss_triplet_margin"]
    func testAllEightLossTypesResolveTheirActualGeneratedImages() throws {
        for id in ids {
            XCTAssertEqual(ComponentArtworkCatalog.assetName(for:id),id+"-v2",id)
            let image=try XCTUnwrap(ComponentArtworkCatalog.image(for:id),id)
            XCTAssertGreaterThan(image.size.width,0);XCTAssertGreaterThan(image.size.height,0)
        }
    }
    func testLossTitlesAndPurposesUseAuthoredPresentationWithoutChangingIt() throws {
        for id in ids {
            let presentation=try XCTUnwrap(BundledLossComponents.entries[id]?.presentation)
            XCTAssertEqual(ComponentStyle.shortTitle(id,fallback:"Unrecognized"),presentation.shortTitle,id)
            XCTAssertEqual(ComponentStyle.purpose(id),presentation.purpose,id)
        }
    }
    func testObjectiveCheckUsesEvaluationSymbolAndFamilyWithoutPretendingToHaveGeneratedArtwork() {
        XCTAssertEqual(ComponentStyle.systemSymbol("objective_check"),"checkmark.seal")
        XCTAssertEqual(ComponentVisualFamily.resolve(type:"objective_check"),.evaluation)
        XCTAssertNil(ComponentArtworkCatalog.assetName(for:"objective_check"))
        XCTAssertNil(ComponentArtworkCatalog.image(for:"objective_check"))
    }
    func testLaterDataArtworkMappingsAreExplicitAndDecode() throws {
        let additions:[String:String]=[
            "kwip.data.sequenceWindows":"data-sequence-windows-v1", "kwip.data.textEncode":"data-text-encoding-v1", "kwip.data.columns":"data-columns-v1", "kwip.data.union":"data-union-v1",
            "kwip.data.filterRows":"data-filter-rows-v1", "kwip.data.partition":"data-split-v2",
            "kwip.data.formula":"data-formula-v1", "kwip.data.preprocess":"data-preprocess-v1",
            "kwip.data.sort":"data-sort-v1", "kwip.data.unique":"data-unique-v1",
            "kwip.data.reshapeTable":"data-reshapeTable-v1", "kwip.data.splitText":"data-splitText-v1"
        ]
        for (id,asset) in additions {
            XCTAssertEqual(ComponentArtworkCatalog.assetName(for:id),asset,id)
            let image=try XCTUnwrap(ComponentArtworkCatalog.image(for:id),id)
            XCTAssertGreaterThan(image.size.width,0);XCTAssertGreaterThan(image.size.height,0)
        }
    }
    func testEveryPriorArtworkMappingIsPreservedExactly() throws {
        // This receipt protects the exact pre-loss map. Later generated families
        // have their own acceptance guards and must not alter this baseline.
        let compositionIDs: Set<String> = ["kwip.composition.sequential", "kwip.composition.orderedList", "kwip.composition.namedMap"]
        let laterIDs: Set<String> = ["kwip.data.sequenceWindows", "kwip.data.textEncode", "embedding_bag", "bag_input", "dataset_source", "data_loader", "program_export", "kwip.data.filterExpression", "kwip.data.calculate", "kwip.data.regexMatch", "kwip.data.regexExtract", "kwip.data.regexReplace", "kwip.data.outputDelimited", "kwip.data.view", "kwip.data.sourceFiles", "kwip.data.columns", "kwip.data.union", "kwip.data.filterRows", "kwip.data.partition", "kwip.data.formula", "kwip.data.preprocess", "kwip.data.sort", "kwip.data.unique", "kwip.data.reshapeTable", "kwip.data.splitText"]
        XCTAssertEqual(ComponentArtworkCatalog.assetName(for:"kwip.data.sourceCSV"),"table-input-v1")
        XCTAssertEqual(ComponentArtworkCatalog.assetName(for:"kwip.data.sourceFiles"),"file-input-v1")
        let historicalMappings=try HistoricalArtworkCompatibility.mapping(ComponentArtworkCatalog.assets)
        let previous=historicalMappings.filter {!laterIDs.contains($0.key) && !ids.contains($0.key) && $0.key != "initialization" && $0.key != "gradient_policy" && $0.key != "trainability" && !compositionIDs.contains($0.key)}
        XCTAssertEqual(previous.count,100)
        let data=Data(previous.keys.sorted().map {$0+"="+previous[$0]!}.joined(separator:"\n").utf8)
        XCTAssertEqual(WorkflowArchiveCodec.hash(data),"3252bc45e0d7cec801f658d368582aaec97d9f581f94d864dec40913d7406739")
        XCTAssertNil(ComponentArtworkCatalog.assetName(for:"uninstalled.plugin"))
    }
}
