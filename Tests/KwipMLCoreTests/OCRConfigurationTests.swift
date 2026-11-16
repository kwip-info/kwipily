import XCTest
@testable import KwipMLCore

final class OCRConfigurationTests:XCTestCase {
    private var root:URL {URL(fileURLWithPath:#filePath).deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()}
    func testPracticeVersionedRoundTripAndExactTablePorts()throws {
        let registry=try DataFlowOperatorRegistry.load(from:root.appendingPathComponent("runner/dataflow-operators.json"))
        let document=try OCRPracticeExample.document()
        XCTAssertEqual(try WorkflowCodec.decode(WorkflowCodec.encode(document)),document)
        let data=try WorkflowDataPreparationPlan.compile(document,registry:registry)
        XCTAssertEqual(data.nodes.count,5)
        let definition=try XCTUnwrap(registry.operatorDefinition(id:OCRConfiguration.operatorID))
        XCTAssertEqual(definition.outputs.map(\.id),["documents","regions"])
        XCTAssertTrue(definition.outputs.allSatisfy {$0.kind == "table"})
        try OCRConfiguration.validate(definition.defaultConfig,definition:definition)
        var bad=definition.defaultConfig;bad["allowlist"] = .string("A");bad["blocklist"] = .string("B")
        XCTAssertThrowsError(try OCRConfiguration.validate(bad,definition:definition))
    }
    func testReviewedDownloadsCannotRequestArbitraryHostFileRangeOrCredential()throws {
        let file=OCRPackCatalog.assets[0]
        let valid=ModelHubRequest(action:.ocrFile,repository:"JaidedAI/EasyOCR",revision:OCRPackCatalog.id,file:file.name,offset:0,total:Int64(file.bytes))
        XCTAssertEqual(try valid.prepared().url?.absoluteString,file.url)
        XCTAssertEqual(try valid.prepared().value(forHTTPHeaderField:"Range"),"bytes=0-1048575")
        for request in [ModelHubRequest(action:.ocrFile,repository:"attacker/model",revision:OCRPackCatalog.id,file:file.name,offset:0,total:Int64(file.bytes)),ModelHubRequest(action:.ocrFile,repository:"JaidedAI/EasyOCR",revision:OCRPackCatalog.id,file:"evil.zip",offset:0,total:100),ModelHubRequest(action:.ocrFile,repository:"JaidedAI/EasyOCR",revision:OCRPackCatalog.id,file:file.name,offset:0,total:Int64(file.bytes),bearer:"hf_private")] {XCTAssertThrowsError(try request.prepared())}
        XCTAssertFalse(ModelHubRequest.allowedRedirect(URL(string:file.url)!))
        XCTAssertFalse(OCRPackCatalog.allowed(URL(string:"https://github.com/attacker/model.zip")!))
    }
    func testLegacyPreparationDefaultsRemainReadableWithoutNewImageOptions()throws {
        var values=try PrepareMediaConfiguration.defaults()
        for key in ["outputMode","cropEnabled","cropX","cropY","cropWidth","cropHeight","rotation"] {values.removeValue(forKey:key)}
        try PrepareMediaConfiguration.validate(values)
        values["rotation"] = 45;XCTAssertThrowsError(try PrepareMediaConfiguration.validate(values))
    }
}
