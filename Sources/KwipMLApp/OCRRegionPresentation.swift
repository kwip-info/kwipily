import Foundation
import KwipMLCore

/// Bounded presentation of the currently loaded standard table page, never an OCR rerun.
struct OCRRegionPresentation:Identifiable {
    let id:Int
    let regionNumber:Int
    let text:String
    let confidence:Double
    let review:String
    let points:[CGPoint]
    static func regions(page:TableViewPage?,image:FileImageReference)->[Self] {
        guard let page,page.rows.count<=512 else {return []}
        let names=page.columns.map {$0["name"]?.stringValue ?? ""}
        func index(_ key:String)->Int? {names.firstIndex(of:key)}
        guard let source=index("source_image"),let text=index("text"),let score=index("confidence"),let review=index("review"),let region=index("region"),let space=index("coordinate_space") else {return []}
        let coordinates=(0..<4).flatMap {[index("x"+String($0)),index("y"+String($0))]}
        guard coordinates.allSatisfy({$0 != nil}) else {return []}
        func number(_ value:JSONValue)->Double? {switch value {case .integer(let n):return Double(n);case .number(let n):return n.isFinite ? n:nil;default:return nil}}
        return page.rows.enumerated().compactMap {position,row in
            guard row.count==names.count,case .object(let reference)=row[source],reference["sha256"] == .string(image.sha256),row[space] == .string("sourceImagePixelsTopLeft"),let content=row[text].stringValue,let confidence=number(row[score]),let status=row[review].stringValue,let id=row[region].integerValue else {return nil}
            let values=coordinates.compactMap {number(row[$0!])}
            guard values.count==8,confidence>=0,confidence<=1 else {return nil}
            return .init(id:position,regionNumber:Int(id),text:content,confidence:confidence,review:status,points:(0..<4).map {CGPoint(x:values[$0*2],y:values[$0*2+1])})
        }
    }
}
