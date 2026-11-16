import XCTest
import KwipMLCore
@testable import KwipMLApp

final class SpeechTranscriptionTests:XCTestCase {
    private var repository:URL {URL(fileURLWithPath:#filePath).deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()}
    func testPracticeHasRealModelResourceEdgeButOnlyTablesEnterPreparationDAG()throws {
        let registry=try DataFlowOperatorRegistry.load(from:repository.appendingPathComponent("runner/dataflow-operators.json"))
        let doc=try SpeechTranscriptionPracticeExample.document()
        let plan=try WorkflowDataPreparationPlan.compile(doc,registry:registry)
        XCTAssertEqual(plan.nodes.count,5);XCTAssertEqual(plan.edges.count,4)
        let node=try XCTUnwrap(doc.nodes.first {$0.address==SpeechTranscriptionPracticeExample.speech})
        let models=try ComponentRegistry.builtIn()
        let ports=try WorkflowPortContracts.resolve(node,document:doc,modelRegistry:models,dataRegistry:registry)
        XCTAssertEqual(ports.inputs.first {$0.id=="model"}?.kind,.frozenModel)
        XCTAssertEqual(ports.inputs.first {$0.id=="model"}?.required,true)
        if case .dataOperator(let id,let config)=node.payload {let definition=try XCTUnwrap(registry.operatorDefinition(id:id));XCTAssertEqual(config,definition.defaultConfig);XCTAssertEqual(definition.inputs.first {$0.id=="model"}?.required,true);try SpeechTranscriptionConfiguration.validate(config,definition:definition)}
        var disconnected=doc;disconnected.edges.removeAll {$0.to.port=="model"}
        XCTAssertThrowsError(try WorkflowDataPreparationPlan.compile(disconnected,registry:registry))
        let encoded=try WorkflowCodec.encode(doc);XCTAssertEqual(try WorkflowCodec.decode(encoded),doc)
        XCTAssertFalse(String(decoding:encoded,as:UTF8.self).contains("/Users/"))
    }
    @MainActor func testLessonLaunchNeedsExplicitSourceAndModelWithoutDownload()throws {
        let root=FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString);defer {try? FileManager.default.removeItem(at:root)}
        let workspace=WorkspaceStore(rootOverride:repository,storageOverride:root,detectCapabilities:false)
        XCTAssertTrue(workspace.loadComponentWorkflowExample(SpeechTranscriptionPracticeExample.exampleID,device:"cpu"),workspace.statusMessage)
        let host=try XCTUnwrap(workspace.workflowSession)
        XCTAssertNil(host.bindings.path(for:SpeechTranscriptionPracticeExample.source));XCTAssertNil(host.modelLoader.reference(SpeechTranscriptionPracticeExample.model))
        XCTAssertEqual(ComponentLessonDestination.entryID(type:SpeechTranscriptionConfiguration.operatorID),SpeechTranscriptionPracticeExample.exampleID)
        let content=try XCTUnwrap(LessonContentCatalog.content(entryID:SpeechTranscriptionPracticeExample.exampleID))
        XCTAssertFalse(content.steps.isEmpty)
    }
    func testLegacyMediaConfigurationDoesNotRequireNewSpectralKeys()throws {
        var legacy=try PrepareMediaConfiguration.defaults(kind:"audio")
        for key in ["representation","fftSize","spectralHop","window","center","melBands","minimumHz","maximumHz","logFloorDB"] {legacy.removeValue(forKey:key)}
        try PrepareMediaConfiguration.validate(legacy)
        legacy["representation"] = .string("logMel")
        XCTAssertThrowsError(try PrepareMediaConfiguration.validate(legacy))
        legacy["sampleFormat"] = .string("float32");try PrepareMediaConfiguration.validate(legacy)
    }
}

extension SpeechTranscriptionTests {
    func testSegmentRangeTravelsWithCapturedRowAndRejectsMalformedTimes()throws {
        let names=["start_seconds","end_seconds","source_audio","timestamp_coverage"]
        let columns=names.enumerated().map{CapturedDataGrid.Column(id:$0.offset,name:$0.element,label:$0.element,dtype:$0.offset<2 ? "float64":"string",nullability:"required")}
        func row(_ a:String,_ b:String)->CapturedDataGrid.Row {.init(index:7,cells:[a,b,"captured","modelSegments"].map{.init(text:$0,isAbsent:false)})}
        XCTAssertEqual(SpeechSegmentPresentation.interval(row:row("1.25","10.5"),columns:columns),1.25...10.5)
        for pair in [("nan","10"),("0","infinity"),("-1","2"),("4","3")] {XCTAssertNil(SpeechSegmentPresentation.interval(row:row(pair.0,pair.1),columns:columns))}
        XCTAssertNil(SpeechSegmentPresentation.interval(row:row("0","1"),columns:Array(columns.prefix(3))))
    }
}

extension SpeechTranscriptionTests {
    func testSpectralPreviewIncludesWholeChannelAndFrequencyRunsUpward()throws {
        func bytes(_ values:[Float])->Data {values.withUnsafeBytes{Data($0)}}
        let a=try SpectralPreview.raster(samples:bytes([0,2,4,6,10,12,14,16]),shape:[2,2,2],channel:1)
        XCTAssertEqual(a.means,[12,16,10,14]);XCTAssertEqual(a.minimum,10);XCTAssertEqual(a.maximum,16)
        let b=try SpectralPreview.raster(samples:bytes((0..<513).map(Float.init)),shape:[1,513,1],channel:0)
        XCTAssertEqual(b.width,512);XCTAssertEqual(b.means.last,511.5);XCTAssertEqual(b.maximum,512)
        XCTAssertThrowsError(try SpectralPreview.raster(samples:bytes([.infinity]),shape:[1,1,1],channel:0))
    }
}
