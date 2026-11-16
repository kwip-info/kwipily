import XCTest
@testable import KwipMLApp

final class PresentationCacheTests:XCTestCase {
    func testStableInputReusesWorkButChangedContentInvalidatesIt() {
        let cache=BoundedPresentationCache<String,String>(maximumEntries:2,maximumCost:20)
        var builds=0
        func resolve(_ value:String)->String {cache.value(key:"lesson",input:value,cost:5) {builds+=1;return value.uppercased()}}
        XCTAssertEqual(resolve("first"),"FIRST")
        XCTAssertEqual(resolve("first"),"FIRST")
        XCTAssertEqual(builds,1)
        XCTAssertEqual(resolve("changed"),"CHANGED")
        XCTAssertEqual(builds,2)
    }
    func testEntryAndCostBudgetsBoundRetentionAndOversizeBypassesCache() {
        let cache=BoundedPresentationCache<Int,Int>(maximumEntries:2,maximumCost:10)
        for i in 0..<100 {_=cache.value(key:String(i),input:i,cost:6) {i}}
        XCTAssertLessThanOrEqual(cache.retainedCount,2)
        XCTAssertLessThanOrEqual(cache.retainedCost,10)
        var builds=0
        for _ in 0..<2 {_=cache.value(key:"huge",input:7,cost:11) {builds+=1;return 7}}
        XCTAssertEqual(builds,2)
        XCTAssertLessThanOrEqual(cache.retainedCost,10)
    }
}

extension PresentationCacheTests {
    func testLessonRevisionStillUsesOriginalDigestAndInvalidatesChangedText() {
        var content=LessonContent(entryID:"cache-check",title:"Lesson",kind:"Practice",steps:[.init(id:"one",title:"Read",blocks:[.init(id:"body",kind:.explanation,title:"",text:"Original")])])
        let original=LessonContent.digest(Data(content.steps.map(\.revision).joined().utf8))
        XCTAssertEqual(content.revision,original)
        content.steps[0].nodeFocus = .init(types:["linear"])
        XCTAssertEqual(content.revision,original,"Presentation references must not reset reading progress")
        content.steps[0].blocks[0].text="Revised"
        XCTAssertNotEqual(content.revision,original)
        XCTAssertEqual(content.revision,LessonContent.digest(Data(content.steps.map(\.revision).joined().utf8)))
    }
    @MainActor func testOptInLessonRevisionCost()throws {
        guard ProcessInfo.processInfo.environment["KWIPML_LESSON_PRESENTATION_BENCHMARK"]=="1" else {throw XCTSkip("Opt-in presentation microbenchmark")}
        let content=try XCTUnwrap(LessonContentCatalog.content(entryID:"train-data-access",operation:nil))
        let expected=content.revision
        let before=Date()
        for _ in 0..<300 {XCTAssertEqual(LessonContent.digest(Data(content.steps.map(\.revision).joined().utf8)),expected)}
        let uncached=Date().timeIntervalSince(before)
        let cachedStart=Date()
        for _ in 0..<300 {XCTAssertEqual(content.revision,expected)}
        print("LESSON_REVISION_COST uncachedSeconds=\(uncached) cachedSeconds=\(Date().timeIntervalSince(cachedStart)) iterations=300 steps=\(content.steps.count)")
    }
}
