import Foundation
import CryptoKit

/// Shared authored presentation. Reading and actions have no assessment authority.
struct LessonContentBlock: Identifiable, Equatable, Codable {
    enum Kind: String, Codable, CaseIterable { case explanation, canvasAction, observation, visual, callout, detail, question, check, feedback }
    var id: String
    var kind: Kind
    var title: String
    var text: String
    var targetType: String? = nil
    var columns: [String] = []
    var rows: [[String]] = []
    var imageName: String? = nil
    var alternativeText: String? = nil
    var answer:String? = nil
    var visual:LessonVisual? = nil
    var choices:[LessonQuestionChoice]? = nil
}
struct LessonContentStep: Identifiable, Equatable, Codable {
    var id: String
    var title: String
    var blocks: [LessonContentBlock]
    var nodeFocus:LessonNodeFocus? = nil
    var canvasFocus:CanvasLessonFocus {(nodeFocus ?? .init(types:blocks.compactMap(\.targetType))).presentation}
    var revision: String { let encoder=JSONEncoder();encoder.outputFormatting = [.sortedKeys];var reading=self;reading.nodeFocus=nil
        for index in reading.blocks.indices {
            if let visual=reading.blocks[index].visual,visual.caption == nil,(visual.definitions ?? []).isEmpty,(visual.nodes ?? []).isEmpty,(visual.links ?? []).isEmpty {reading.blocks[index].visual=nil}
        }
        return LessonContent.digest((try? encoder.encode(reading)) ?? Data()) }
}
struct LessonContent: Equatable, Codable {
    var entryID: String
    var title: String
    var kind: String
    var steps: [LessonContentStep]
    /// Stable task identity for reading only; bookmarks and assessed receipts retain their own keys.
    var readingScopeID:String? = nil
    private static let revisions=BoundedPresentationCache<[LessonContentStep],String>(maximumEntries:32,maximumCost:2_000_000)
    var revision: String {
        let cost=steps.reduce(0) {$0+256+$1.blocks.reduce(0) {$0+256+$1.text.utf8.count*4}}
        return Self.revisions.value(key:entryID,input:steps,cost:cost) {Self.digest(Data(steps.map(\.revision).joined().utf8))}
    }
    static func digest(_ data: Data) -> String { SHA256.hash(data: data).map { String(format: "%02x", $0) }.joined() }
    static func step(_ id: String, _ title: String, _ text: String, kind: LessonContentBlock.Kind = .explanation, target: String? = nil) -> LessonContentStep {
        .init(id:id,title:title,blocks:[.init(id:id+".body",kind:kind,title:"",text:text,targetType:target)])
    }
}

/// Opt-in authored presentation. This metadata never executes code or awards correctness.
struct LessonQuestionChoice:Codable,Equatable,Identifiable {var id:String;var text:String}
struct LessonVisual:Codable,Equatable {
    enum Kind:String,Codable,CaseIterable {case table,matrix,equation,diagram,image,code}
    var kind:Kind
    var caption:String? = nil
    var definitions:[LessonVisualDefinition]? = nil
    var nodes:[LessonDiagramNode]? = nil
    var links:[LessonDiagramLink]? = nil
}
struct LessonVisualDefinition:Codable,Equatable {var symbol:String;var meaning:String}
struct LessonDiagramNode:Codable,Equatable,Identifiable {var id:String;var title:String;var subtitle:String? = nil;var emphasized:Bool? = nil}
struct LessonDiagramLink:Codable,Equatable {var from:String;var to:String}
