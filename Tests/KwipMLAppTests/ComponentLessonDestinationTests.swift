import XCTest
import KwipMLCore
@testable import KwipMLApp

@MainActor final class ComponentLessonDestinationTests: XCTestCase {
    func testDocumentTaskLinksAndLegacyDefaultsRemainDistinct() {
        XCTAssertEqual(ComponentLessonDestination.entryID(type:"estimator",configuration:DocumentClassificationConfiguration.defaults),DocumentClassificationPracticeExample.exampleID)
        XCTAssertEqual(ComponentLessonDestination.entryID(type:"kwip.data.textEncode",configuration:["mode":.string("documentFeatures")]),DocumentClassificationPracticeExample.exampleID)
        XCTAssertEqual(ComponentLessonDestination.entryID(type:"frozenModel",configuration:["source":.string("documentPackage")]),DocumentClassificationPracticeExample.exampleID)
        XCTAssertEqual(ComponentLessonDestination.entryID(type:"estimator"),"sklearn")
        XCTAssertEqual(ComponentLessonDestination.entryID(type:"frozenModel"),"load-model")
        XCTAssertEqual(ComponentLessonDestination.entryID(type:"kwip.data.textEncode"),TextEncodingPracticeExample.lessonID)
    }
    func testBoundaryGuideAndDedicatedDepthwiseLessonAreDiscoverable() {
        XCTAssertEqual(ComponentLessonDestination.entryID(type:"depthwise_conv2d"),"depthwise")
        XCTAssertEqual(ComponentLessonDestination.entryID(type:"input"),"model-input")
        XCTAssertEqual(ComponentLessonDestination.entryID(type:"output"),"model-output")
        XCTAssertTrue(ComponentLessonDestination.isAvailable(entryID:"depthwise"))
        XCTAssertTrue(ComponentLessonDestination.isAvailable(entryID:"guided"))
    }
    func testActualToolsReachExistingLessonRatherThanLabelSearch() throws {
        let cases = ["kwip.data.preprocess":"preprocess", "kwip.data.imputerFit":"preprocess", "kwip.data.missing":"preprocess", "linear":"affine", "relu":"relu-lesson", "kwip.data.join":"join", "train":"train-data-access", "dataset_source":"train-data-access"]
        for (type, expected) in cases {
            XCTAssertEqual(ComponentLessonDestination.entryID(type:type), expected)
            XCTAssertTrue(LearnCatalogContent.items.contains {$0.id == expected} || LearnCatalogContent.historicalEntryIDs.contains(expected))
        }
        XCTAssertEqual(ComponentLessonDestination.entryID(type:"estimator",configuration:["backendID":.string("tensorflow")]),"tensorflow")
        XCTAssertNil(ComponentLessonDestination.entryID(type:"unregistered_custom_tool"))
        XCTAssertNil(ComponentLessonDestination.entryID(type:"estimator",configuration:["backendID":.string("unknown")]))
    }
    func testWorkflowDisplayTypeUsesQualifiedIdentityNotAuthoredName() throws {
        let doc=WorkflowDocument(projectID:UUID().uuidString,nodes:[.init(address:.init(domain:.model,localID:"renamed"),name:"Not a lesson title",x:0,y:0,payload:.modelPrimitive(type:"linear",configurationVersion:2,config:["in_features":2,"out_features":2,"bias":true]))],edges:[])
        let projection=try WorkflowCanvasProjector.project(doc,modelRegistry:.builtIn(),dataRegistry:DataFlowOperatorRegistry.load(from:URL(fileURLWithPath:#filePath).deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent().appendingPathComponent("runner/dataflow-operators.json")))
        let node=try XCTUnwrap(projection.graph.nodes.first)
        XCTAssertEqual(ComponentLessonDestination.entryID(type:node.type),"affine")
    }
    func testDataControlsBundlesAndCustomBlocksUseExactAvailableReadingDestinations() {
        let cases = ["kwip.data.preprocess":"preprocess", "kwip.data.imputerFit":"preprocess", "kwip.data.missing":"preprocess", "kwip.data.calculate": "formula", "kwip.data.formula":"formula", "kwip.data.regexExtract":"formula", "kwip.data.regexReplace":"formula",
                     "kwip.data.cast": "data-preparation-course", "gradient_policy": "train-data-access",
                     "initialization": "train-data-access", "data_loader": "train-data-access",
                     "composite:personal-block@1.0.0": "module-composition", "optimizer_adam": "train-data-access"]
        for (type, entry) in cases {
            XCTAssertEqual(ComponentLessonDestination.entryID(type: type), entry)
            XCTAssertTrue(ComponentLessonDestination.isAvailable(entryID: entry))
        }
        for type in ["not_a_component", "wf-t-corrupt", "kwip.data.fake", "kwip.data.regexMatch", "composite:"] {
            XCTAssertNil(ComponentLessonDestination.entryID(type: type))
        }
    }
    func testConsolidatedPracticeOperatorsShareDedicatedLessonsWithoutAnAmbiguousLookup() throws {
        let lessons=try DataEngineeringPractice.catalog().lessons
        let groups=Dictionary(grouping:lessons,by: \.operatorID)
        XCTAssertTrue(groups.values.contains {$0.count>1},"Exercise consolidation deliberately reuses tool identities")
        let destinations=["kwip.data.filterRows":"filter","kwip.data.formula":"formula","kwip.data.outputDelimited":"table-output"]
        XCTAssertEqual(Set(groups.keys),Set(destinations.keys))
        for lesson in lessons {
            XCTAssertEqual(ComponentLessonDestination.entryID(type:lesson.operatorID),destinations[lesson.operatorID])
        }
        XCTAssertEqual(ComponentLessonDestination.entryID(type:"kwip.data.selectRename"),"data-preparation-course")
        XCTAssertNil(ComponentLessonDestination.entryID(type:"unregistered_custom_tool"))
    }

    func testDirectNavigationIsExactRejectsUnknownAndSearchLeavesReadingMode() {
        let navigation = LearningLibraryNavigation()
        XCTAssertTrue(navigation.open(entryID: "affine"))
        XCTAssertEqual(navigation.section, "Library")
        XCTAssertEqual(navigation.search, "")
        XCTAssertEqual(navigation.directEntryID, "affine")
        XCTAssertFalse(navigation.open(entryID: "not-a-lesson"))
        XCTAssertEqual(navigation.directEntryID, "affine")
        navigation.open(search: "diagnostics")
        XCTAssertNil(navigation.directEntryID)
        XCTAssertEqual(navigation.search, "diagnostics")
    }

    func testTaskConfigurationSharesNativeAndMCPDestinations()throws {
        for (id,expected) in [("self-play","self-play"),("fine-tuning","fine-tuning"),("typed-decisions","typed-decisions")] {
            let document=try PreparedWorkflowExample.component(id).document(device:"cpu")
            let payload=try XCTUnwrap(document.nodes.first {if case .train = $0.payload{return true};return false}?.payload)
            guard case .train(let configuration)=payload else{return XCTFail("Expected Train")}
            XCTAssertEqual(ComponentLessonDestination.entryID(payload:payload),expected)
            XCTAssertEqual(ComponentLessonDestination.entryID(type:"workflow.train",configuration:configuration),expected)
        }
        XCTAssertEqual(ComponentLessonDestination.entryID(type:"mixture_experts"),"mixture-experts")
        XCTAssertEqual(ComponentLessonDestination.entryID(type:"output",configuration:["task":.string("gamePolicyValue")]),"self-play")
        for type in ["workflow.train","workflow.frozenModel","workflow.saveModel","workflow.logisticRegression"] {
            for id in ComponentLessonDestination.relatedEntryIDs(type:type) {
                XCTAssertTrue(ComponentLessonDestination.isAvailable(entryID:id),type+" "+id)
            }
        }
    }
    func testExplicitHubProfilesRouteByTaskButLocalPathsNeverGuessSupport() {
        let profiles=["openai/whisper-tiny":"transcribe-audio","MIT/ast-finetuned-audioset-10-10-0.4593":"classify-audio"]
        for (repository,lesson) in profiles {
            XCTAssertEqual(ComponentLessonDestination.entryID(type:"frozenModel",configuration:["source":.string("huggingFace"),"repository":.string(repository)]),lesson)
            XCTAssertEqual(ComponentLessonDestination.entryID(type:"frozenModel",configuration:["source":.string("localFolder"),"repository":.string(repository)]),"load-model")
        }
        XCTAssertEqual(ComponentLessonDestination.entryID(type:"frozenModel",configuration:["source":.string("huggingFace"),"repository":.string("convaiinnovations/laya"),"subfolder":.string("typed-decisions")]),"typed-decisions")
    }

}
