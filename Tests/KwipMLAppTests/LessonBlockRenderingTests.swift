import XCTest
import SwiftUI
import AppKit
import KwipMLCore
@testable import KwipMLApp

@MainActor final class LessonBlockRenderingTests:XCTestCase {
    func testNumberedDirectionsDoNotMergeOrMistakeDecimalValuesForSteps() {
        let blocks=LessonTextLayout.blocks("1. Find **Model Output**.\n2. Use 1.0 exactly.\n3. Inspect [N, 2].\n\n1.0 is a value, not a step.\n\n```\n2. a literal code line\n```")
        XCTAssertEqual(blocks.map(\.kind),[.numbered(1),.numbered(2),.numbered(3),.paragraph,.code])
        XCTAssertEqual(blocks[1].text,"Use 1.0 exactly.")
        XCTAssertEqual(blocks.last?.text,"2. a literal code line")
    }
    func testVisualTableIsBoundedAndKeepsBlankCellsRatherThanShiftingColumns() {
        let projection=LessonVisualTable(columns:["a","b","c"],rows:[["1"],["2","3","4","extra"]]+Array(repeating:["5"],count:90))
        XCTAssertEqual(projection.rows.count,64)
        XCTAssertEqual(projection.rows[0],["1","",""])
        XCTAssertEqual(projection.rows[1],["2","3","4"])
        XCTAssertEqual(projection.totalRows,92)
        XCTAssertTrue(projection.isBounded)
    }
    func testOptionalVisualAndReflectionMetadataRoundTripsAndRealContentChangesReadingRevision() throws {
        var step=LessonContent.step("x","Predict","Which shape preserves the batch?")
        let previous=step.revision
        step.blocks[0].visual = .init(kind:.equation)
        XCTAssertEqual(step.revision,previous,"A style hint must not invalidate existing reading")
        step.blocks[0].visual = .init(kind:.equation,caption:"Authored relationship")
        step.blocks[0].choices = [.init(id:"a",text:"[N, 2]")]
        XCTAssertNotEqual(step.revision,previous)
        XCTAssertEqual(try JSONDecoder().decode(LessonContentStep.self,from:JSONEncoder().encode(step)),step)
    }
    func testTwoNodeDiagramsFitTheInspectorWithoutClippingTheSecondNode() {
        for width:CGFloat in [270,300,360] {
            let layout=LessonDiagramLayout(count:2,availableWidth:width)
            XCTAssertLessThanOrEqual(layout.width+10,width)
            XCTAssertGreaterThanOrEqual(layout.nodeWidth,92)
        }
    }
    func testAssessmentFeedbackDoesNotExposeFormativeEvidence() {
        let outcome=LearningCriterionOutcome(criterionID:"shape",passed:false,evidence:"Private fixture values",nextStep:"Review dimensions")
        let assessment=LessonFeedbackItem.project(outcome,title:"Output shape",publicNextStep:"Check sample dimensions",assessment:true)
        XCTAssertNil(assessment.evidence)
        XCTAssertEqual(assessment.nextStep,"Check sample dimensions")
        XCTAssertEqual(LessonFeedbackItem.project(outcome,title:"Output shape",assessment:false).evidence,outcome.evidence)
        XCTAssertNil(LessonBundledImage.image("../private.png"))
    }
    func testEveryBlockAndVisualRendersAtNarrowWidthInBothThemesWithoutExecutingActions() async throws {
        _=NSApplication.shared
        var actions=0
        let blocks=Self.samples
        XCTAssertEqual(Set(blocks.map(\.kind)),Set(LessonContentBlock.Kind.allCases))
        XCTAssertEqual(Set(blocks.compactMap {$0.visual?.kind}),Set(LessonVisual.Kind.allCases))
        for (scheme,size) in [(ColorScheme.dark,12),(.light,12),(.dark,24),(.light,24)] {
            let theme=AppearanceThemePolicy(preferences:.factory,colorScheme:scheme,increasedContrast:false,reduceTransparency:false)
            let content=VStack(alignment:.leading,spacing:12) {
                ForEach(blocks) {block in
                    LessonContentBlockView(block:block,selectTarget:{_ in actions+=1},check:{actions+=1})
                }
            }.padding(14).frame(width:300).background(theme.backgroundColor)
                .environment(\.colorScheme,scheme).environment(\.appearanceTheme,theme)
                .environment(\.lessonTypography,LessonTypography(bodySize:size))
                .buttonStyle(FlatButtonStyle())
            let host=NSHostingView(rootView:content)
            let window=NSWindow(contentRect:NSRect(x:0,y:0,width:300,height:4000),styleMask:.borderless,backing:.buffered,defer:false)
            window.isReleasedWhenClosed=false;window.appearance=NSAppearance(named:scheme == .dark ? .darkAqua:.aqua)
            window.contentView=host
            host.layoutSubtreeIfNeeded();await Task.yield();host.layoutSubtreeIfNeeded()
            let size=host.fittingSize
            XCTAssertEqual(size.width,300,accuracy:1)
            XCTAssertGreaterThan(size.height,1000)
            XCTAssertLessThan(size.height,7000)
            host.frame=NSRect(origin:.zero,size:size)
            host.layoutSubtreeIfNeeded();await Task.yield();host.layoutSubtreeIfNeeded()
            if let directory=ProcessInfo.processInfo.environment["KWIPML_LESSON_RENDER_ARTIFACTS"] {
                try FileManager.default.createDirectory(atPath:directory,withIntermediateDirectories:true)
                let bitmap=try XCTUnwrap(host.bitmapImageRepForCachingDisplay(in:host.bounds))
                host.cacheDisplay(in:host.bounds,to:bitmap)
                let data=try XCTUnwrap(bitmap.representation(using:.png,properties:[:]))
                try data.write(to:URL(fileURLWithPath:directory).appendingPathComponent("blocks-"+(scheme == .dark ? "dark":"light")+"-\(size).png"))
            }
            window.contentView=nil;window.close()
        }
        XCTAssertEqual(actions,0,"Rendering and reading must never submit or select automatically")
    }
    private static var samples:[LessonContentBlock] {
        var blocks=LessonContentBlock.Kind.allCases.map {kind in
            LessonContentBlock(id:kind.rawValue,kind:kind,title:"",text:kind == .canvasAction ? "1. Find **Model Output**.\n2. Inspect sample shape `[2]`.":"A batch groups observations. Each observation keeps the same **sample shape**.",targetType:kind == .canvasAction ? "output":nil)
        }
        if let index=blocks.firstIndex(where:{$0.kind == .question}) {
            blocks[index].choices=[.init(id:"a",text:"`[N, 2]`"),.init(id:"b",text:"`[N, 3]`")]
            blocks[index].answer="The leading dimension may vary."
        }
        for kind in LessonVisual.Kind.allCases {
            var block=LessonContentBlock(id:"visual."+kind.rawValue,kind:.visual,title:kind.rawValue.capitalized,text:kind == .code ? "# Read-only relationship\ndef forward(x):\n    return x":"y = xWᵀ + b",columns:["Feature A","Feature B","Feature C"],rows:[["1","2","3"],["4","5","6"]])
            block.visual = .init(kind:kind,caption:"Authored example",definitions:kind == .equation ? [.init(symbol:"W",meaning:"Learned weights")]:nil,nodes:kind == .diagram ? [.init(id:"a",title:"Class scores",subtitle:"Tensor [N, 2]",emphasized:true),.init(id:"b",title:"Model Output",emphasized:true)]:nil,links:kind == .diagram ? [.init(from:"a",to:"b")]:nil)
            if kind == .image {block.imageName="linear-v1.png";block.alternativeText="Layer artwork"}
            blocks.append(block)
        }
        return blocks
    }
    func testUnavailableCancelledStaleAndUnsavedGradesNeverLookLikeNewVerifiedSuccess() {
        let grade=LearningGrade(status:.passed,summary:"The independent checks passed.")
        XCTAssertEqual(LessonCheckState(grade:nil,running:false,canSubmit:false,unavailable:true,current:false).label,"Unavailable")
        XCTAssertEqual(LessonCheckState(grade:.init(status:.cancelled,summary:"Stopped"),running:false,canSubmit:true,unavailable:false,current:true).label,"Cancelled")
        XCTAssertEqual(LessonCheckState(grade:grade,running:false,canSubmit:true,unavailable:false,current:false).label,"Previous result")
        XCTAssertEqual(LessonCheckState(grade:grade,running:true,canSubmit:false,unavailable:false,current:true).label,"Checking")
    }
}
