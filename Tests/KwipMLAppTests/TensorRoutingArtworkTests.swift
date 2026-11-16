import AppKit
import CryptoKit
import XCTest
import KwipMLCore
@testable import KwipMLApp

@MainActor final class TensorRoutingArtworkTests: XCTestCase {
    private let laterDataAssets:[String:String]=[
        "kwip.data.sequenceWindows":"data-sequence-windows-v1", "kwip.data.textEncode":"data-text-encoding-v1", "kwip.data.columns":"data-columns-v1", "kwip.data.union":"data-union-v1",
        "kwip.data.filterRows":"data-filter-rows-v1", "kwip.data.partition":"data-split-v2",
        "kwip.data.formula":"data-formula-v1", "kwip.data.preprocess":"data-preprocess-v1",
        "kwip.data.sort":"data-sort-v1", "kwip.data.unique":"data-unique-v1",
        "kwip.data.reshapeTable":"data-reshapeTable-v1", "kwip.data.splitText":"data-splitText-v1"
    ]
    private let members:[(id:String,title:String,purpose:String)]=[
        ("split","Split","Separate an array into ordered parts with chosen sizes."),
        ("chunk","Chunk","Request ordered parts and inspect how many actually exist."),
        ("gather","Gather","Choose source values by their exact positions."),
        ("einsum","Einsum","Combine or rearrange arrays using named arrangements.")]

    func testEveryRoutingComponentDecodesItsDistinctOriginalTransparentArtwork() throws {
        for member in members {
            XCTAssertEqual(ComponentArtworkCatalog.assetName(for:member.id),member.id+"-v2")
            let image=try XCTUnwrap(ComponentArtworkCatalog.image(for:member.id),member.id)
            let bitmap=try XCTUnwrap(NSBitmapImageRep(data:try XCTUnwrap(image.tiffRepresentation)),member.id)
            XCTAssertTrue(bitmap.hasAlpha,member.id)
            XCTAssertGreaterThan(bitmap.pixelsWide,0)
            XCTAssertGreaterThan(bitmap.pixelsHigh,0)
        }
    }

    func testReadableSharedLabelsAndPurpleRectangularFamilyMatchAuthoredMetadata() throws {
        let registry=try ComponentRegistry.builtIn()
        for member in members {
            let definition=try XCTUnwrap(registry.component(type:member.id),member.id)
            XCTAssertEqual(definition.title,member.title)
            XCTAssertEqual(definition.category,"Tensor layout")
            XCTAssertEqual(ComponentStyle.shortTitle(member.id,fallback:"Unlabelled component"),member.title)
            XCTAssertEqual(ComponentStyle.purpose(member.id),member.purpose)
            let family=ComponentVisualFamily.resolve(type:member.id)
            XCTAssertEqual(family,.tensorLayout)
            XCTAssertEqual(family.colors.primary,0xB3A3D4)
            XCTAssertEqual(family.colors.secondary,0x76638F)
            XCTAssertEqual(family.silhouette,.rectangle)
        }
    }

    func testLaterDataAssetsHaveExactInventoryMappingsAndDecode() throws {
        let root=URL(fileURLWithPath:#filePath).deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
        let catalog=try XCTUnwrap(JSONSerialization.jsonObject(with:Data(contentsOf:root.appendingPathComponent("docs/design/component-icons-v2/catalog.json"))) as? [String:Any])
        let entries=try XCTUnwrap(catalog["components"] as? [[String:String]])
        for (id,asset) in laterDataAssets {
            let rows=entries.filter {$0["id"]==id}
            XCTAssertEqual(rows.count,1,id)
            XCTAssertEqual(try XCTUnwrap(rows.first)["asset"],asset+".png",id)
            XCTAssertEqual(ComponentArtworkCatalog.assetName(for:id),asset,id)
            let image=try XCTUnwrap(ComponentArtworkCatalog.image(for:id),id)
            XCTAssertGreaterThan(image.size.width,0);XCTAssertGreaterThan(image.size.height,0)
        }
    }

    func testCatalogRecordsCanonicalNewAssetsAndPreservesAllPrior67Mappings() throws {
        let root=URL(fileURLWithPath:#filePath).deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
        let catalog=try XCTUnwrap(JSONSerialization.jsonObject(with:Data(contentsOf:root.appendingPathComponent("docs/design/component-icons-v2/catalog.json"))) as? [String:Any])
        let entries=try XCTUnwrap(catalog["components"] as? [[String:String]])
        for member in members {
            let matching=entries.filter {$0["id"]==member.id}
            XCTAssertEqual(matching.count,1,member.id)
            XCTAssertEqual(try XCTUnwrap(matching.first),["id":member.id,"title":member.title,"category":"Tensor layout","asset":member.id+"-v2.png"])
        }
        let ids=Set(members.map(\.id)).union(["unfold","fold","batch_norm1d","batch_norm2d","instance_norm1d","instance_norm2d","group_norm","optimizer_sgd","optimizer_adam","optimizer_adamw","optimizer_rmsprop","optimizer_adagrad","scheduler_step_lr","scheduler_cosine_annealing_lr","scheduler_reduce_on_plateau","scheduler_one_cycle_lr","scheduler_linear_lr","rnn","lstm","gru","rnn_cell","lstm_cell","gru_cell","multihead_attention","transformer","transformer_encoder","transformer_decoder","scaled_dot_product_attention","attention_mask","loss_cross_entropy","loss_bce_with_logits","loss_mse","loss_l1","loss_nll","loss_kl_div","loss_ctc","loss_triplet_margin","initialization","gradient_policy","trainability","kwip.composition.sequential","kwip.composition.orderedList","kwip.composition.namedMap"])
        let laterIDs: Set<String> = Set(["embedding_bag", "bag_input", "dataset_source", "data_loader", "program_export", "kwip.data.filterExpression", "kwip.data.calculate", "kwip.data.regexMatch", "kwip.data.regexExtract", "kwip.data.regexReplace", "kwip.data.outputDelimited", "kwip.data.view", "kwip.data.sourceFiles"]).union(laterDataAssets.keys)
        XCTAssertEqual(ComponentArtworkCatalog.assetName(for:"kwip.data.sourceCSV"),"table-input-v1")
        let historicalMappings=try HistoricalArtworkCompatibility.mapping(ComponentArtworkCatalog.assets)
        let old=historicalMappings.filter {!ids.contains($0.key) && !laterIDs.contains($0.key)}
        XCTAssertEqual(old.count,67)
        let bytes=try JSONSerialization.data(withJSONObject:old,options:[.sortedKeys,.withoutEscapingSlashes])
        // Foundation's sorted-key ordering, independently serialized from commit 830adff.
        XCTAssertEqual(SHA256.hash(data:bytes).map {String(format:"%02x",$0)}.joined(),"f84a59dfe8f65b60901ae8b64cec068bb1ea517907aeb0df258ec8498e58025b")
        let tableInputRows=entries.filter {$0["id"] == "kwip.data.sourceCSV"}
        XCTAssertEqual(tableInputRows.count,1)
        let tableInput=try XCTUnwrap(tableInputRows.first)
        XCTAssertEqual(tableInput,["id":"kwip.data.sourceCSV","title":"Table Input","category":"Data","asset":"table-input-v1.png"])
        XCTAssertEqual(ComponentArtworkCatalog.assetName(for:"kwip.data.sourceCSV"),"table-input-v1")
        // The reviewed Table Input alias corrects metadata only. Preserve the
        // original entry proof by normalizing this exact asserted row, not others.
        let currentOutput=try XCTUnwrap(entries.first {$0["id"] == "output"})
        XCTAssertEqual(currentOutput,["id":"output","title":"Model Output","category":"Boundaries","asset":"model-boundary-v3.png"])
        let legacyOutput=["id":"output","title":"Output","category":"Data","asset":"output-v2.png"]
        let oldEntries=entries.filter {!ids.contains($0["id"] ?? "") && !laterIDs.contains($0["id"] ?? "") && historicalMappings[$0["id"] ?? ""] != nil}.map { entry in
            var projected=entry
            if let asset=historicalMappings[entry["id"] ?? ""] {projected["asset"]=asset+".png"}
            if entry["id"]=="input" || entry["id"]=="index_input" {
                XCTAssertEqual(entry["title"],"Model input");XCTAssertEqual(entry["category"],"Boundaries")
                projected["title"]=entry["id"]=="input" ? "Input" : "Integer input"
                projected["category"]=entry["id"]=="input" ? "Data" : "Boundaries"
            }
            if entry["id"]=="workflow.frozenModel" {
                XCTAssertEqual(entry,["id":"workflow.frozenModel","title":"Load model","category":"Boundaries","asset":"frozen-model-v2.png","rendering":"native"])
                projected.removeValue(forKey:"rendering");projected["title"]="Frozen model";projected["category"]="Evaluation"
            }
            if entry["id"]=="dropout" {XCTAssertEqual(entry["category"],"Conditioning");projected["category"]="Activations"}
            return entry == tableInput ? ["id":"kwip.data.sourceCSV","title":"CSV source","category":"Data","asset":"csv-table-v2.png"] : entry == currentOutput ? legacyOutput : projected
        }
        XCTAssertEqual(oldEntries.count,67)
        let entryBytes=try JSONSerialization.data(withJSONObject:oldEntries,options:[.sortedKeys,.withoutEscapingSlashes])
        XCTAssertEqual(SHA256.hash(data:entryBytes).map {String(format:"%02x",$0)}.joined(),"78c74847ec0e55f32a2ccfe60a1eada9b88b757e090b2f9d2ab1a83d0a0d8009")
    }
}
