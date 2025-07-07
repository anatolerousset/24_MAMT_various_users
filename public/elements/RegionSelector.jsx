import { useState } from 'react';
import { Card, CardContent } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { 
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { MapPin, Database, Check } from "lucide-react";

export default function RegionSelector() {
  const [selectedRegion, setSelectedRegion] = useState(props.selected_region || '');
  const [isUpdating, setIsUpdating] = useState(false);

  const handleRegionChange = (value) => {
    setSelectedRegion(value);
  };

  const handleConfirmSelection = async () => {
    if (!selectedRegion) return;
    
    setIsUpdating(true);
    
    try {
      // Update the element props with the selected region
      await updateElement({
        ...props,
        selected_region: selectedRegion,
        collection_name: `region_${selectedRegion.toLowerCase()}_documents`,
        last_updated: new Date().toISOString()
      });
      
      // Call action to notify the backend about the region change
      await callAction({
        name: 'region_selected',
        payload: {
          region: selectedRegion,
          collection_name: `region_${selectedRegion.toLowerCase()}_documents`
        }
      });
      
    } catch (error) {
      console.error('Error updating region:', error);
    } finally {
      setIsUpdating(false);
    }
  };

  const availableRegions = props.available_regions || [];
  const currentCollection = selectedRegion ? `region_${selectedRegion.toLowerCase()}_documents` : props.default_collection || 'region_paca_documents';

  return (
    <Card className="w-full max-w-md mx-auto">
      <CardContent className="p-6">
        <div className="space-y-4">
          {/* Header */}
          <div className="flex items-center gap-2">
            <MapPin className="h-5 w-5 text-blue-600" />
            <h3 className="text-lg font-semibold text-gray-900">
              Sélection de Région
            </h3>
          </div>
          
          {/* Description */}
          <p className="text-sm text-gray-600">
            Choisissez la région pour la recherche dans les mémoires techniques
          </p>
          
          {/* Region Selector */}
          <div className="space-y-2">
            <label className="text-sm font-medium text-gray-700">
              Région disponible:
            </label>
            <Select value={selectedRegion} onValueChange={handleRegionChange}>
              <SelectTrigger className="w-full">
                <SelectValue placeholder="Sélectionnez une région..." />
              </SelectTrigger>
              <SelectContent>
                {availableRegions.map((region) => (
                  <SelectItem key={region} value={region}>
                    <div className="flex items-center gap-2">
                      <MapPin className="h-4 w-4" />
                      {region}
                    </div>
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>
          </div>

          {/* Collection Info */}
          {selectedRegion && (
            <div className="bg-blue-50 border border-blue-200 rounded-lg p-3">
              <div className="flex items-center gap-2 text-blue-800">
                <Database className="h-4 w-4" />
                <span className="text-sm font-medium">Collection technique:</span>
              </div>
              <p className="text-sm text-blue-700 mt-1 font-mono">
                {currentCollection}
              </p>
            </div>
          )}

          {/* Confirm Button */}
          <Button 
            onClick={handleConfirmSelection}
            disabled={!selectedRegion || isUpdating}
            className="w-full"
            variant={selectedRegion ? "default" : "secondary"}
          >
            {isUpdating ? (
              <div className="flex items-center gap-2">
                <div className="animate-spin rounded-full h-4 w-4 border-b-2 border-white"></div>
                Mise à jour...
              </div>
            ) : (
              <div className="flex items-center gap-2">
                <Check className="h-4 w-4" />
                Confirmer la sélection
              </div>
            )}
          </Button>

          {/* Current Status */}
          {props.selected_region && (
            <div className="text-center">
              <p className="text-xs text-gray-500">
                Région actuelle: <span className="font-semibold text-gray-700">{props.selected_region}</span>
              </p>
              {props.last_updated && (
                <p className="text-xs text-gray-400">
                  Dernière mise à jour: {new Date(props.last_updated).toLocaleString('fr-FR')}
                </p>
              )}
            </div>
          )}
        </div>
      </CardContent>
    </Card>
  );
}