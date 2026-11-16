import SwiftUI
import KwipMLCore

struct DataFlowNumberField<Context: DataNodeEditing>: View {
    @ObservedObject var context: Context
    let key: String
    let title: String
    let domain: NumericDomain
    let fallback: NumericValue
    private var value: NumericValue {
        let raw = context.node?.config[key]
        switch raw {
        case .integer(let number): return .integer(number)
        case .number(let number): if case .integer = domain { return .wholeDisplay(number) }; return .decimal(number)
        default: return fallback
        }
    }
    var body: some View {
        let origin = context.numericOrigin(key:key)
        let expected = context.node?.config
        BoundedNumberField(title:title,value:value,domain:domain,origin:origin,currentOrigin:{
            context.currentNumericOrigin(key:key)
        },onDraftChange:{ originating,pending in
            context.setNumericPending(origin:originating,key:key,pending:pending)
        }) { value in
            guard var config=expected else {return false}
            switch value {case .integer(let number):config[key] = .integer(number);case .decimal(let number):config[key] = .number(number)}
            return context.configure(config,expected:expected ?? [:])
        }.id(origin)
    }
}
