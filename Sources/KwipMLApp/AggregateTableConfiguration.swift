import Foundation
import KwipMLCore

/// Declared editing guidance, independent of execution's row and partition validation.
enum AggregateTableConfiguration {
    static func validate(_ config:[String:JSONValue],schema:[DataColumnSchema])throws->[String:JSONValue] {
        guard Set(config.keys) == ["groupBy","nullKeys","aggregations","maxRows"],case .array(let groups)=config["groupBy"],case .array(let reducers)=config["aggregations"],!reducers.isEmpty,groups.count+reducers.count <= 128,case .string(let nullKeys)=config["nullKeys"],["group","drop","error"].contains(nullKeys),case .integer(let maxRows)=config["maxRows"],(1...100000).contains(maxRows) else {throw DataFlowContractError("Choose ordered grouping columns, at least one reducer, a missing-key policy and a row limit from 1 to 100,000. Output is limited to 128 columns.")}
        func name(_ value:JSONValue?)throws->String {
            guard case .string(let result)=value,!result.trimmingCharacters(in:.whitespacesAndNewlines).isEmpty,result.count <= 200 else {throw DataFlowContractError("Enter an exact nonblank name of at most 200 characters.")};return result
        }
        var outputNames=Set<String>()
        for group in groups {
            let key=try name(group)
            guard outputNames.insert(key).inserted,schema.isEmpty || schema.contains(where:{$0.name == key && $0.dtype != "image"}) else {throw DataFlowContractError("Grouping columns must be unique and present in the recorded input schema.")}
        }
        for item in reducers {
            guard case .object(let reducer)=item,case .string(let op)=reducer["op"],["countRows","countNonNull","sum","sumDecimal","mean","min","max","joinText"].contains(op) else {throw DataFlowContractError("Choose a supported reducer.")}
            let output=try name(reducer["output"])
            guard outputNames.insert(output).inserted else {throw DataFlowContractError("Output names must be unique, including grouping columns.")}
            var required:Set<String> = op == "countRows" ? ["output","op"]:op == "countNonNull" ? ["output","op","column"]:["output","op","column","nullPolicy"]
            if op == "joinText" {required.formUnion(["separator","distinct","skipEmpty"])}
            guard Set(reducer.keys) == required else {throw DataFlowContractError("Reducer settings do not match its operation.")}
            if op == "joinText" {
                guard case .string(let separator)=reducer["separator"],separator.unicodeScalars.count <= 1024,case .bool=reducer["distinct"],case .bool=reducer["skipEmpty"] else {throw DataFlowContractError("Join text requires a separator of at most 1,024 characters and duplicate/empty choices.")}
            }
            if op != "countRows" {
                let column=try name(reducer["column"]),known=schema.first {$0.name == column}
                guard schema.isEmpty || known != nil else {throw DataFlowContractError("Reducer column is absent from the recorded input schema.")}
                if known?.dtype == "image" {throw DataFlowContractError("Image columns cannot be summarized as scalar values.")}
                if op == "sumDecimal",let known,known.dtype != "string" {throw DataFlowContractError("Exact decimal sum requires preserved decimal Text, not a Number column.")}
                if op == "joinText",let known,known.dtype != "string" {throw DataFlowContractError("Join text requires a Text column. Use Columns for explicit casts.")}
                if ["sum","mean"].contains(op),let known,!["int64","float64"].contains(known.dtype) {throw DataFlowContractError("Sum and mean require a numeric column. Use Columns for explicit casts.")}
            }
            if !["countRows","countNonNull"].contains(op) {
                guard case .string(let policy)=reducer["nullPolicy"],["skip","propagate","error"].contains(policy) else {throw DataFlowContractError("Choose the reducer's missing-value policy.")}
            }
        }
        return config
    }
    static func outputSchema(_ config:[String:JSONValue],schema:[DataColumnSchema])throws->[DataColumnSchema] {
        _=try validate(config,schema:schema)
        guard !schema.isEmpty,case .array(let groups)=config["groupBy"],case .array(let reducers)=config["aggregations"] else {return []}
        var result=groups.compactMap { value -> DataColumnSchema? in
            guard case .string(let name)=value,var column=schema.first(where:{$0.name == name}) else {return nil}
            if config["nullKeys"] != .string("group") {column.nullable=false};return column
        }
        for item in reducers {
            guard case .object(let reducer)=item,case .string(let output)=reducer["output"],case .string(let op)=reducer["op"] else {continue}
            if ["countRows","countNonNull"].contains(op) {result.append(.init(name:output,dtype:"int64",nullable:false));continue}
            guard case .string(let name)=reducer["column"],let column=schema.first(where:{$0.name == name}) else {continue}
            result.append(.init(name:output,dtype:op == "mean" ? "float64":column.dtype,nullable:reducer["nullPolicy"] == .string("error") ? false:column.nullable))
        }
        return result
    }
}
