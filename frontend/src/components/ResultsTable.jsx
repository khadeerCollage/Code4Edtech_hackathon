import { useEffect, useMemo, useState } from "react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card.jsx";
import { Table, TableBody, TableCaption, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table.jsx";
import { Input } from "@/components/ui/input.jsx";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select.jsx";
import { cn } from "@/lib/utils.js";
import { X } from "lucide-react";

const ResultsTable = ({ title = "Results", data }) => {
  const [query, setQuery] = useState("");
  const [column, setColumn] = useState("name"); // all | name | math | physics | chemistry | history | total
  const [debouncedQuery, setDebouncedQuery] = useState("");

  useEffect(() => {
    const t = setTimeout(() => setDebouncedQuery(query), 250);
    return () => clearTimeout(t);
  }, [query]);

  const filtered = useMemo(() => {
    const q = debouncedQuery.trim().toLowerCase();
    if (!q) return data;
    return data.filter((r) => {
      const total = r.math + r.physics + r.chemistry + r.history;
      switch (column) {
        case "name":
          return r.name.toLowerCase().includes(q);
        case "math":
          return String(r.math).toLowerCase().includes(q);
        case "physics":
          return String(r.physics).toLowerCase().includes(q);
        case "chemistry":
          return String(r.chemistry).toLowerCase().includes(q);
        case "history":
          return String(r.history).toLowerCase().includes(q);
        case "total":
          return String(total).toLowerCase().includes(q);
        case "all":
        default:
          return (
            r.name.toLowerCase().includes(q) ||
            String(r.math).includes(q) ||
            String(r.physics).includes(q) ||
            String(r.chemistry).includes(q) ||
            String(r.history).includes(q) ||
            String(total).includes(q)
          );
      }
    });
  }, [data, debouncedQuery, column]);

  return (
    <Card className="shadow-lg animate-slide-up overflow-hidden">
      <CardHeader className="pb-2 space-y-3">
        <CardTitle className="text-base md:text-lg">{title}</CardTitle>
        <div className="grid grid-cols-10 gap-2 w-full items-stretch">
          <div className="relative col-span-7">
            <Input
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              placeholder="Search by name, subject marks, or total..."
              aria-label="Search results"
              className="h-10 pr-10 transition-all duration-200 focus:shadow-glow animate-fade-in"
            />
            {query && (
              <button
                type="button"
                aria-label="Clear search"
                onClick={() => setQuery("")}
                className="absolute right-2 top-1/2 -translate-y-1/2 text-muted-foreground hover:text-foreground transition-colors"
              >
                <X className="w-4 h-4" />
              </button>
            )}
          </div>
          <div className="col-span-3">
            <Select value={column} onValueChange={setColumn}>
              <SelectTrigger className="h-10 w-full transition-all duration-200 focus:shadow-glow animate-slide-up">
                <SelectValue placeholder="Filter column" />
              </SelectTrigger>
              <SelectContent className={cn("animate-in zoom-in-95 fade-in-0", "data-[state=closed]:animate-out data-[state=closed]:fade-out-0 data-[state=closed]:zoom-out-95")}>
                <SelectItem value="all">All columns</SelectItem>
                <SelectItem value="name">Name</SelectItem>
                <SelectItem value="math">Math</SelectItem>
                <SelectItem value="physics">Physics</SelectItem>
                <SelectItem value="chemistry">Chemistry</SelectItem>
                <SelectItem value="history">History</SelectItem>
                <SelectItem value="total">Total</SelectItem>
              </SelectContent>
            </Select>
          </div>
        </div>
      </CardHeader>
      <CardContent>
        <div className="rounded-xl border bg-white/90 overflow-x-auto">
          <Table className="text-sm">
            <TableHeader>
              <TableRow className="bg-blue-600/90 text-white">
                <TableHead className="text-white">Student Name</TableHead>
                <TableHead className="text-white">Section</TableHead>
                <TableHead className="text-white">Math</TableHead>
                <TableHead className="text-white">Physics</TableHead>
                <TableHead className="text-white">Chemistry</TableHead>
                <TableHead className="text-white">History</TableHead>
                <TableHead className="text-white">Total Result</TableHead>
              </TableRow>
            </TableHeader>
            <TableBody>
              {filtered.map((r, i) => {
                const total = r.math + r.physics + r.chemistry + r.history;
                return (
                  <TableRow key={`${r.name}-${i}`}>
                    <TableCell className="text-foreground">{r.name}</TableCell>
                    <TableCell className="text-foreground">{r.section}</TableCell>
                    <TableCell className="text-foreground">{r.math}</TableCell>
                    <TableCell className="text-foreground">{r.physics}</TableCell>
                    <TableCell className="text-foreground">{r.chemistry}</TableCell>
                    <TableCell className="text-foreground">{r.history}</TableCell>
                    <TableCell>
                      <span className="text-blue-600 font-medium">{total}</span>
                    </TableCell>
                  </TableRow>
                );
              })}
            </TableBody>
            {filtered.length === 0 && <TableCaption className="text-xs">No results match your search</TableCaption>}
          </Table>
        </div>
      </CardContent>
    </Card>
  );
};

export default ResultsTable;
