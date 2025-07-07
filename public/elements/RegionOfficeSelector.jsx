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
import { MapPin, Building2, Database, Check } from "lucide-react";

export default function RegionOfficeSelector() {
  const [selectedRegion, setSelectedRegion] = useState(props.selected_region || '');
  const [selectedOffice, setSelectedOffice] = useState(props.selected_office || '');
  const [isUpdating, setIsUpdating] = useState(false);

  const handleRegionChange = (value) => {
    setSelectedRegion(value);
  };

  const handleOfficeChange = (value) => {
    setSelectedOffice(value);
  };

  const handleConfirmSelection = async () => {
    if (!selectedRegion || !selectedOffice) return;
    
    setIsUpdating(true);
    
    try {
      // Update the element props with the selected region and office
      await updateElement({
        ...props,
        selected_region: selectedRegion,
        selected_office: selectedOffice,
        technical_collection_name: `region_${selectedRegion.toLowerCase()}_documents`,
        dce_collection_name: `dce_${selectedOffice.toLowerCase()}_documents`,
        last_updated: new Date().toISOString()
      });
      
      // Call action to notify the backend about the selection change
      await callAction({
        name: 'region_office_selected',
        payload: {
          region: selectedRegion,
          office: selectedOffice,
          technical_collection_name: `region_${selectedRegion.toLowerCase()}_documents`,
          dce_collection_name: `dce_${selectedOffice.toLowerCase()}_documents`
        }
      });
      
    } catch (error) {
      console.error('Error updating region/office:', error);
    } finally {
      setIsUpdating(false);
    }
  };

  const availableRegions = props.available_regions || [];
  const availableOffices = props.available_offices || [];
  const currentTechnicalCollection = selectedRegion ? `region_${selectedRegion.toLowerCase()}_documents` : props.default_technical_collection || 'region_paca_documents';
  const currentDceCollection = selectedOffice ? `dce_${selectedOffice.toLowerCase()}_documents` : props.default_dce_collection || 'dce_documents';

  const isSelectionComplete = selectedRegion && selectedOffice;

  return (
    <Card className="w-full max-w-lg mx-auto">
      <CardContent className="p-6">
        <div className="space-y-4">
          {/* Header */}
          <div className="flex items-center gap-2">
            <div className="flex items-center gap-1">
              <MapPin className="h-5 w-5 text-blue-600" />
              <Building2 className="h-5 w-5 text-green-600" />
            </div>
            <h3 className="text-lg font-semibold text-gray-900">
              Sélection Région & Bureau
            </h3>
          </div>
          
          {/* Description */}
          <p className="text-sm text-gray-600">
            Choisissez la région (mémoires techniques) et le bureau (DCE) pour configurer les collections appropriées
          </p>
          
          {/* Region Selector */}
          <div className="space-y-2">
            <label className="text-sm font-medium text-gray-700 flex items-center gap-2">
              <MapPin className="h-4 w-4 text-blue-600" />
              Région (Collection Technique):
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

          {/* Office Selector */}
          <div className="space-y-2">
            <label className="text-sm font-medium text-gray-700 flex items-center gap-2">
              <Building2 className="h-4 w-4 text-green-600" />
              Bureau (Collection DCE):
            </label>
            <Select value={selectedOffice} onValueChange={handleOfficeChange}>
              <SelectTrigger className="w-full">
                <SelectValue placeholder="Sélectionnez un bureau..." />
              </SelectTrigger>
              <SelectContent>
                {availableOffices.map((office) => (
                  <SelectItem key={office} value={office}>
                    <div className="flex items-center gap-2">
                      <Building2 className="h-4 w-4" />
                      {office}
                    </div>
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>
          </div>

          {/* Collections Info */}
          {isSelectionComplete && (
            <div className="space-y-3">
              <div className="bg-blue-50 border border-blue-200 rounded-lg p-3">
                <div className="flex items-center gap-2 text-blue-800">
                  <Database className="h-4 w-4" />
                  <span className="text-sm font-medium">Collection technique:</span>
                </div>
                <p className="text-sm text-blue-700 mt-1 font-mono">
                  {currentTechnicalCollection}
                </p>
              </div>
              
              <div className="bg-green-50 border border-green-200 rounded-lg p-3">
                <div className="flex items-center gap-2 text-green-800">
                  <Database className="h-4 w-4" />
                  <span className="text-sm font-medium">Collection DCE:</span>
                </div>
                <p className="text-sm text-green-700 mt-1 font-mono">
                  {currentDceCollection}
                </p>
              </div>
            </div>
          )}

          {/* Warning if incomplete selection */}
          {!isSelectionComplete && (selectedRegion || selectedOffice) && (
            <div className="bg-amber-50 border border-amber-200 rounded-lg p-3">
              <div className="flex items-center gap-2 text-amber-800">
                <div className="w-4 h-4 rounded-full border-2 border-amber-600 flex items-center justify-center">
                  <div className="w-1 h-1 bg-amber-600 rounded-full"></div>
                </div>
                <span className="text-sm font-medium">Sélection incomplète</span>
              </div>
              <p className="text-sm text-amber-700 mt-1">
                Veuillez sélectionner à la fois une région et un bureau
              </p>
            </div>
          )}

          {/* Confirm Button */}
          <Button 
            onClick={handleConfirmSelection}
            disabled={!isSelectionComplete || isUpdating}
            className="w-full"
            variant={isSelectionComplete ? "default" : "secondary"}
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
          {(props.selected_region || props.selected_office) && (
            <div className="text-center space-y-1">
              {props.selected_region && (
                <p className="text-xs text-gray-500">
                  Région actuelle: <span className="font-semibold text-blue-700">{props.selected_region}</span>
                </p>
              )}
              {props.selected_office && (
                <p className="text-xs text-gray-500">
                  Bureau actuel: <span className="font-semibold text-green-700">{props.selected_office}</span>
                </p>
              )}
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