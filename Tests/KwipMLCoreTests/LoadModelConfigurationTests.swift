import XCTest
@testable import KwipMLCore

final class LoadModelConfigurationTests:XCTestCase {
    func testLegacySelectionStaysEmptyAndExternalSettingsRoundTrip()throws {
        XCTAssertEqual(try LoadModelConfiguration.decode([:]).source,.savedRun)
        XCTAssertTrue(LoadModelConfiguration().values.isEmpty)
        let value=LoadModelConfiguration(source:.localFolder,maximumTokens:32)
        XCTAssertEqual(try LoadModelConfiguration.decode(value.values),value)
        var bad=value.values;bad["path"] = .string("/private/model")
        XCTAssertThrowsError(try LoadModelConfiguration.decode(bad))
        bad=value.values;bad["maximumTokens"]=true
        XCTAssertThrowsError(try LoadModelConfiguration.decode(bad))
    }
    func testVersionedExternalSourcePreservesLegacyFrozenNode()throws {
        var doc=WorkflowDocument(projectID:UUID().uuidString,nodes:[.init(address:.init(domain:.control,localID:"loader"),payload:.frozenModel(config:[:]))])
        doc.schemaVersion=19
        XCTAssertEqual(try WorkflowCodec.decode(WorkflowCodec.encode(doc)),doc)
        doc.schemaVersion=20
        doc.nodes[0].payload = .frozenModel(config:LoadModelConfiguration(source:.localFolder).values)
        XCTAssertEqual(try WorkflowCodec.decode(WorkflowCodec.encode(doc)),doc)
    }
    func testSaveFormatOptionsAreStrictAndLegacyDefaultsRemain()throws {
        XCTAssertEqual(try WorkflowSaveModelConfiguration.decode([:]).format,.package)
        let value=WorkflowSaveModelConfiguration(format:.huggingFace,shardBytes:100_000_000)
        XCTAssertEqual(try WorkflowSaveModelConfiguration.decode(value.values),value)
        XCTAssertThrowsError(try WorkflowSaveModelConfiguration.decode(["format":.string("package"),"shardBytes":100_000_000]))
        XCTAssertThrowsError(try WorkflowSaveModelConfiguration.decode(["shardBytes":100_000_000]))
        XCTAssertThrowsError(try WorkflowSaveModelConfiguration.decode(["format":.string("huggingFace"),"shardBytes":0]))
    }
    func testHubHasFinitePinnedRangesAndNoArbitraryTargets()throws {
        let sha=String(repeating:"a",count:40)
        let chunk=ModelHubRequest(action:.file,repository:"org/model",revision:sha,file:"model.safetensors",offset:0,total:2_000_000)
        let request=try chunk.prepared()
        XCTAssertEqual(request.value(forHTTPHeaderField:"Range"),"bytes=0-1048575")
        XCTAssertEqual(request.url?.host,"huggingface.co")
        XCTAssertThrowsError(try ModelHubRequest(action:.file,repository:"org/model",revision:"main",file:"config.json",offset:0,total:12).prepared())
        XCTAssertThrowsError(try ModelHubRequest(action:.resolve,repository:"../model",revision:"main").prepared())
        for name in ["../secret","model.py","pytorch_model.bin","sub/../config.json","https://evil.test/file"] {XCTAssertFalse(ModelHubRequest.allowedFile(name))}
        XCTAssertFalse(ModelHubRequest.allowedRedirect(URL(string:"https://huggingface.co.evil.test/file")!))
        XCTAssertFalse(ModelHubRequest.allowedRedirect(URL(string:"http://huggingface.co/file")!))
        for host in ["us.aws.cdn.hf.co","us.gcp.cdn.hf.co"] {
            XCTAssertTrue(ModelHubRequest.allowedRedirect(URL(string:"https://"+host+"/weights")!))
            XCTAssertFalse(ModelHubRequest.allowedRedirect(URL(string:"https://"+host+".evil.test/weights")!))
        }
    }
}
