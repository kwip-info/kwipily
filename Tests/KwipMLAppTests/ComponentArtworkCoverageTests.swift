import XCTest
import AppKit
import KwipMLCore
@testable import KwipMLApp

@MainActor final class ComponentArtworkCoverageTests:XCTestCase {
    func testAllShippedCatalogTypesHaveDecodableGeneratedArtwork() throws {
        let root=URL(fileURLWithPath:#filePath).deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
        struct Catalog:Decodable {struct Entry:Decodable {let id:String;let asset:String;let rendering:String?};let components:[Entry]}
        let catalog=try JSONDecoder().decode(Catalog.self,from:Data(contentsOf:root.appendingPathComponent("docs/design/component-icons-v2/catalog.json")))
        let ids=Set(catalog.components.map(\.id))
        XCTAssertEqual(ids.count,catalog.components.count)
        for component in try ComponentRegistry.builtIn().components {
            XCTAssertTrue(ids.contains(component.id),"Missing inventory: \(component.id)")
            XCTAssertNotNil(ComponentArtworkCatalog.image(for:component.id),"Registered component has no decoded generated artwork: \(component.id)")
        }
        for operation in try DataFlowOperatorRegistry.load(from:root.appendingPathComponent("runner/dataflow-operators.json")).operators where operation.id != DataFlowSourceOperators.capturedTable {XCTAssertTrue(ids.contains(operation.id),"Missing data inventory: \(operation.id)")}
        for entry in catalog.components {
            if entry.rendering == "native" {
                XCTAssertEqual(entry.id,"workflow.frozenModel","Native inventory exceptions are explicit")
                XCTAssertNil(ComponentArtworkCatalog.assetName(for:entry.id))
                let symbol=try XCTUnwrap(ComponentStyle.systemSymbol(entry.id))
                XCTAssertNotNil(NSImage(systemSymbolName:symbol,accessibilityDescription:nil))
                continue
            }
            XCTAssertEqual(URL(fileURLWithPath:entry.asset).pathExtension,"png","Inventory artwork must be a PNG: \(entry.id)")
            XCTAssertEqual(ComponentArtworkCatalog.assetName(for:entry.id),URL(fileURLWithPath:entry.asset).deletingPathExtension().lastPathComponent,"Inventory must name the artwork actually registered: \(entry.id)")
            XCTAssertNotNil(ComponentArtworkCatalog.image(for:entry.id),"Missing generated artwork: \(entry.id)")
        }
    }
    func testBackendTrainersHaveDistinctArtworkIdentity() {
        XCTAssertEqual(WorkflowCanvasAdapter.icon(.estimator(config:["backendID":.string("sklearn")])),"workflow.logisticRegression")
        XCTAssertEqual(WorkflowCanvasAdapter.icon(.estimator(config:["backendID":.string("tensorflow")])),"workflow.kerasClassifier")
        XCTAssertEqual(WorkflowCanvasAdapter.icon(.train(config:[:])),"workflow.train")
    }
    func testCustomBlocksUseGeneratedBlockButUnknownTypesKeepFallback() {
        XCTAssertNotNil(ComponentArtworkCatalog.image(for:"composite:owner.architecture@1.0.0"))
        XCTAssertNil(ComponentArtworkCatalog.assetName(for:"uninstalled.plugin"))
    }
}
